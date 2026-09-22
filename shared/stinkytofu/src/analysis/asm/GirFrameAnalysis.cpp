/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"

#include <algorithm>
#include <charconv>
#include <deque>
#include <limits>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {
namespace {

uint64_t edgeKey(const BasicBlock* from, const BasicBlock* to) {
    const auto a = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(from));
    const auto b = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(to));
    return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6U) + (a >> 2U));
}

int parseInt(const std::string& text, const char* what) {
    int value = 0;
    auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size())
        report_fatal_error(std::string("GIR frame contract: invalid ") + what + " '" + text + "'");
    return value;
}

uint64_t parseU64(const std::string& text, const char* what) {
    uint64_t value = 0;
    auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size())
        report_fatal_error(std::string("GIR frame contract: invalid ") + what + " '" + text + "'");
    return value;
}


//: One record's `key=value` words. A bare word is a flag, stored with an empty value.
using ContractFields = std::map<std::string, std::string>;

ContractFields parseFields(const std::vector<std::string>& words, size_t first,
                           const std::string& line) {
    ContractFields out;
    for (size_t i = first; i < words.size(); ++i) {
        const size_t split = words[i].find('=');
        const std::string key =
            split == std::string::npos ? words[i] : words[i].substr(0, split);
        const std::string value = split == std::string::npos ? "" : words[i].substr(split + 1);
        if (!out.emplace(key, value).second)
            report_fatal_error("GIR frame contract: duplicate field '" + key + "' in '" + line +
                               "'");
    }
    return out;
}


int fieldInt(const ContractFields& fields, const char* key, const std::string& line) {
    auto it = fields.find(key);
    if (it == fields.end())
        report_fatal_error(std::string("GIR frame contract: '") + line + "' needs a " + key + "=");
    return parseInt(it->second, key);
}

int fieldInt(const ContractFields& fields, const char* key, const std::string&, int fallback) {
    auto it = fields.find(key);
    return it == fields.end() ? fallback : parseInt(it->second, key);
}

uint64_t fieldU64(const ContractFields& fields, const char* key, const std::string& line) {
    auto it = fields.find(key);
    if (it == fields.end())
        report_fatal_error(std::string("GIR frame contract: '") + line + "' needs a " + key + "=");
    return parseU64(it->second, key);
}


//: Strip a `#` comment and surrounding blanks; an empty result is a line with nothing on it.
std::string contractLine(const std::string& raw) {
    std::string text = raw.substr(0, raw.find('#'));
    const size_t first = text.find_first_not_of(" \t\r");
    if (first == std::string::npos) return {};
    return text.substr(first, text.find_last_not_of(" \t\r") - first + 1);
}

std::vector<std::string> words(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream in(line);
    std::string word;
    while (in >> word) out.push_back(word);
    return out;
}


GirHazardKind parseHazardKind(const std::string& kind) {
    if (kind == "RAW") return GirHazardKind::RAW;
    if (kind == "WAR") return GirHazardKind::WAR;
    if (kind == "WAW") return GirHazardKind::WAW;
    report_fatal_error("GIR frame contract: unknown hazard kind '" + kind + "'");
}

const Loop* containingLoop(const std::vector<Loop>& loops, const BasicBlock* block) {
    const Loop* best = nullptr;
    for (const Loop& loop : loops) {
        if (!loop.contains(block)) continue;
        if (best == nullptr || loop.bodyBBs.size() < best->bodyBBs.size()) best = &loop;
    }
    return best;
}

bool instructionRealizes(GirActionKind kind, const StinkyInstruction& inst) {
    if (kind == GirActionKind::Read) return isDSRead(inst);
    if (kind == GirActionKind::Copy) return isTensorLoad(inst) || isDSWrite(inst);
    return false;
}

GirFrame advanceFrame(const GirFrame& input, BasicBlock* from, BasicBlock* to,
                      const std::vector<Loop>& loops,
                      const std::unordered_map<int, const Loop*>& genLoops,
                      const GirFrameContract& contract) {
    GirFrame result = input;
    for (const auto& [genId, loop] : genLoops) {
        const GirGenerationSpec& gen = contract.generations.at(genId);
        // Any edge into the header from INSIDE the loop is a back edge: keying on the latch alone
        // made a second back edge take the entry arm and RESET the phase instead of advancing it.
        const bool backEdge = to == loop->headerBB ? loop->contains(from) : from == loop->latchBB;
        if (backEdge)
            result.setPhase(genId, (result.phaseOf(genId) + gen.advance) % gen.ring);
        else if (to == loop->headerBB)
            result.setPhase(genId, gen.entry % gen.ring);
    }
    return result;
}

// A region of -1 is the whole operand, so it meets every region of the same operand.
bool disjointRegions(const GirAccessSpec& a, const GirAccessSpec& b) {
    if (a.operand != b.operand) return true;
    return a.region >= 0 && b.region >= 0 && a.region != b.region;
}

bool wawOrderedByRead(const GirAccessSpec& producer, const GirAccessSpec& consumer,
                      const GirFrameContract& contract) {
    if (producer.ring <= 1) return false;
    for (const GirAccessSpec& access : contract.accesses)
        if (!access.isWrite && !disjointRegions(producer, access) &&
            !disjointRegions(consumer, access))
            return true;
    return false;
}

using StorageKey = std::tuple<std::string, int, int>;

// (operand, region, generation) names exactly one slot, so a touch has exactly one storage id.
int concreteStorage(const GirAccessSpec& access, int generation,
                    std::map<StorageKey, int>& storageIds) {
    StorageKey key{access.operand, access.region, generation};
    return storageIds.emplace(std::move(key), static_cast<int>(storageIds.size())).first->second;
}

struct StaticTouchKey {
    StinkyInstruction* inst = nullptr;
    size_t accessIndex = 0;

    bool operator==(const StaticTouchKey&) const = default;
    bool operator<(const StaticTouchKey& other) const {
        if (inst != other.inst) return std::less<StinkyInstruction*>{}(inst, other.inst);
        return accessIndex < other.accessIndex;
    }
};

struct LiveTouchKey {
    GirFrame frame;
    StaticTouchKey touch;

    bool operator==(const LiveTouchKey&) const = default;
    bool operator<(const LiveTouchKey& other) const {
        if (!(frame == other.frame)) return frame < other.frame;
        return touch < other.touch;
    }
};

struct SlotLive {
    std::map<LiveTouchKey, int> writes;
    std::map<LiveTouchKey, int> reads;
};

using SlotState = std::map<int, SlotLive>;
using HazardState = std::map<GirFrame, SlotState>;
using OccurrenceRegistry = std::map<LiveTouchKey, const GirAccessOccurrence*>;
using OccurrencesByNode =
    std::unordered_map<GirFrameNode, std::vector<const GirAccessOccurrence*>, GirFrameNodeHash>;

using HazardEmitter =
    std::function<void(const GirAccessOccurrence&, const GirAccessOccurrence&, int)>;

HazardState walkHazards(const GirFrameNode& node,
                        const std::vector<const GirAccessOccurrence*>& touches, HazardState state,
                        const GirFrameContract& contract, const OccurrenceRegistry& registry,
                        const HazardEmitter* emit = nullptr) {
    for (const GirAccessOccurrence* touch : touches) {
        const GirAccessSpec& spec = contract.accesses.at(touch->accessIndex);
        const LiveTouchKey me{node.frame, {touch->inst, touch->accessIndex}};
        const int storage = touch->storage;
        if (emit) {
            for (const auto& [storedFrame, slots] : state) {
                (void)storedFrame;
                auto slot = slots.find(storage);
                if (slot == slots.end()) continue;
                if (spec.isWrite)
                    for (const auto& [prior, distance] : slot->second.reads) {
                        auto occurrence = registry.find(prior);
                        if (occurrence == registry.end())
                            report_fatal_error(
                                "GirFrameHazardAnalysis lost a live reader occurrence");
                        (*emit)(*occurrence->second, *touch, distance);
                    }
                for (const auto& [prior, distance] : slot->second.writes) {
                    auto occurrence = registry.find(prior);
                    if (occurrence == registry.end())
                        report_fatal_error(
                            "GirFrameHazardAnalysis lost a live writer occurrence");
                    (*emit)(*occurrence->second, *touch, distance);
                }
            }
        }

        if (spec.isWrite) {
            // A write kills this absolute storage slot in every frame picture.
            for (auto& [storedFrame, slots] : state) {
                (void)storedFrame;
                slots.erase(storage);
            }
            SlotLive replacement;
            replacement.writes.emplace(me, 0);
            state[node.frame][storage] = std::move(replacement);
        } else {
            bool filed = false;
            for (auto& [storedFrame, slots] : state) {
                (void)storedFrame;
                auto slot = slots.find(storage);
                if (slot == slots.end()) continue;
                slot->second.reads[me] = 0;
                filed = true;
            }
            if (!filed) state[node.frame][storage].reads[me] = 0;
        }
    }
    return state;
}

HazardState stepHazards(HazardState state) {
    for (auto& [frame, slots] : state) {
        (void)frame;
        for (auto& [storage, live] : slots) {
            (void)storage;
            for (auto& [touch, distance] : live.writes) {
                (void)touch;
                ++distance;
            }
            for (auto& [touch, distance] : live.reads) {
                (void)touch;
                ++distance;
            }
        }
    }
    return state;
}

template <typename Map>
bool mergeNearest(Map& destination, const Map& source) {
    bool changed = false;
    for (const auto& [key, distance] : source) {
        auto found = destination.find(key);
        if (found == destination.end() || distance < found->second) {
            destination[key] = distance;
            changed = true;
        }
    }
    return changed;
}

bool mergeHazardState(HazardState& destination, const HazardState& source) {
    bool changed = false;
    for (const auto& [frame, slots] : source)
        for (const auto& [storage, live] : slots) {
            SlotLive& target = destination[frame][storage];
            changed |= mergeNearest(target.writes, live.writes);
            changed |= mergeNearest(target.reads, live.reads);
        }
    return changed;
}

std::unordered_map<GirFrameNode, HazardState, GirFrameNodeHash> reachingHazards(
    const GirFrameAnalysis::Result& frames, const OccurrencesByNode& occurrences,
    const OccurrenceRegistry& registry) {
    std::unordered_map<GirFrameNode, HazardState, GirFrameNodeHash> incoming;
    std::deque<GirFrameNode> work;
    for (const auto& [node, successors] : frames.edges) {
        (void)successors;
        incoming.emplace(node, HazardState{});
        work.push_back(node);
    }

    while (!work.empty()) {
        GirFrameNode node = std::move(work.back());
        work.pop_back();
        auto touches = occurrences.find(node);
        static const std::vector<const GirAccessOccurrence*> empty;
        HazardState aged =
            stepHazards(walkHazards(node, touches == occurrences.end() ? empty : touches->second,
                                    incoming[node], frames.contract, registry));
        auto successors = frames.edges.find(node);
        if (successors == frames.edges.end()) continue;
        for (const GirFrameNode& successor : successors->second) {
            auto target = incoming.find(successor);
            if (target == incoming.end()) continue;
            if (mergeHazardState(target->second, aged)) work.push_back(successor);
        }
    }
    return incoming;
}

uint64_t actionOf(const StinkyInstruction* inst) {
    const GirActionData* data = inst ? inst->getModifier<GirActionData>() : nullptr;
    return data ? data->actionId : 0;
}

}  // namespace

GirFrameContract GirFrameContract::parse(const std::string& text) {
    GirFrameContract result;
    if (text.empty()) return result;

    std::istringstream input(text);
    std::string raw;
    bool sawMarker = false;
    while (std::getline(input, raw)) {
        const std::string line = contractLine(raw);
        if (line.empty()) continue;
        if (!sawMarker) {
            if (line != kGirFrameContractMarker)
                report_fatal_error("GIR frame contract: expected '" +
                                   std::string(kGirFrameContractMarker) + "', got '" + line + "'");
            sawMarker = true;
            result.loaded = true;
            continue;
        }
        const std::vector<std::string> field = words(line);
        const std::string& tag = field.front();
        if (tag == "gen") {
            const ContractFields fields = parseFields(field, 2, line);
            GirGenerationSpec gen{parseInt(field.at(1), "gen id"), fieldInt(fields, "ring", line),
                                  fieldInt(fields, "entry", line, 0),
                                  fieldInt(fields, "advance", line, 0)};
            if (gen.ring < 1) report_fatal_error("GIR frame contract: ring must be positive");
            result.generations[gen.id] = gen;
        } else if (tag == "incoming" || tag == "transfer") {
            const bool relative = tag == "transfer";
            const ContractFields fields = parseFields(field, 1, line);
            result.incomings.push_back({fieldU64(fields, "dst", line),
                                        fieldU64(fields, "src", line),
                                        fieldInt(fields, "gen", line),
                                        fieldInt(fields, relative ? "delta" : "value", line),
                                        relative});
        } else if (tag == "rel") {
            const ContractFields fields = parseFields(field, 3, line);
            result.relations.push_back({parseU64(field.at(1), "fence action"),
                                        parseHazardKind(field.at(2)),
                                        fieldU64(fields, "producer", line),
                                        fieldU64(fields, "consumer", line),
                                        fieldInt(fields, "gap", line, 0)});
        } else {
            report_fatal_error("GIR frame contract: unknown record '" + tag + "'");
        }
    }
    if (!sawMarker)
        report_fatal_error("GIR frame contract: missing '" +
                           std::string(kGirFrameContractMarker) + "' marker");
    return result;
}

int GirFrame::phaseOf(int genId) const {
    auto it = std::lower_bound(phases.begin(), phases.end(), genId,
                               [](const auto& item, int id) { return item.first < id; });
    return it != phases.end() && it->first == genId ? it->second : 0;
}

void GirFrame::setPhase(int genId, int phase) {
    auto it = std::lower_bound(phases.begin(), phases.end(), genId,
                               [](const auto& item, int id) { return item.first < id; });
    if (it != phases.end() && it->first == genId)
        it->second = phase;
    else
        phases.insert(it, {genId, phase});
}

size_t GirFrameHash::operator()(const GirFrame& frame) const {
    size_t result = 0;
    for (const auto& [gen, phase] : frame.phases)
        result ^= std::hash<int>{}(gen) ^
                  (std::hash<int>{}(phase) + 0x9e3779b9U + (result << 6U) + (result >> 2U));
    return result;
}

size_t GirFrameNodeHash::operator()(const GirFrameNode& node) const {
    size_t result = std::hash<const BasicBlock*>{}(node.block);
    result ^= GirFrameHash{}(node.frame) + 0x9e3779b9U + (result << 6U) + (result >> 2U);
    return result ^ (std::hash<uint64_t>{}(node.incomingAction) + 0x9e3779b9U + (result << 6U) +
                     (result >> 2U));
}

bool GirFrameAnalysis::Result::isBackEdge(const BasicBlock* from, const BasicBlock* to) const {
    return backEdges.contains(edgeKey(from, to));
}

const std::vector<GirFrame>& GirFrameAnalysis::Result::frames(const BasicBlock* block) const {
    static const std::vector<GirFrame> empty;
    auto it = blockFrames.find(const_cast<BasicBlock*>(block));
    return it == blockFrames.end() ? empty : it->second;
}

GirFrameAnalysis::Result GirFrameAnalysis::run(Function& function, AnalysisManager& AM) {
    Result result;
    const auto encoded = function.getStringMetaData(kGirFrameContractKey);
    if (!encoded || encoded->empty()) return result;
    result.contract = GirFrameContract::parse(*encoded);

    std::unordered_map<StinkyInstruction*, size_t> instructionIndex;
    std::unordered_map<StinkyInstruction*, BasicBlock*> instructionBlock;
    std::unordered_map<BasicBlock*, uint64_t> blockAnchor;
    std::unordered_map<BasicBlock*, uint64_t> blockAnchorOrder;
    std::unordered_map<BasicBlock*, std::unordered_set<uint64_t>> blockActions;
    std::unordered_map<uint64_t, std::unordered_set<BasicBlock*>> anchorBlocks;
    std::unordered_set<uint64_t> realizedAnchors;
    for (BasicBlock& block : function) {
        size_t index = 0;
        for (IRBase& node : block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            instructionIndex[inst] = index++;
            instructionBlock[inst] = &block;
            if (const auto* action = inst->getModifier<GirActionData>()) {
                result.actionInstructions[action->actionId].push_back(inst);
                blockActions[&block].insert(action->actionId);
                // The action and what it touches are facts ABOUT THIS INSTRUCTION, so they come
                // off the modifier.  Legalization can split one rocisa instruction into several
                // carrying the same id, so only the first occurrence contributes its accesses.
                auto [spec, fresh] = result.contract.actions.try_emplace(
                    action->actionId,
                    GirActionSpec{action->actionId, action->anchorAction, action->kind, {}});
                if (fresh) {
                    for (const GirAccessData& touch : action->accesses) {
                        if (touch.ring < 1)
                            report_fatal_error("GirFrameAnalysis: access ring must be positive");
                        spec->second.accesses.push_back(result.contract.accesses.size());
                        result.contract.accesses.push_back(
                            GirAccessSpec{action->actionId, touch.isWrite, touch.genId, touch.ring,
                                          touch.gdelta, touch.absolute, touch.crossAgent,
                                          touch.operand, touch.region});
                    }
                }
                const uint64_t canonical = spec->second.anchorAction;
                realizedAnchors.insert(canonical);
                anchorBlocks[canonical].insert(&block);
                auto [order, inserted] = blockAnchorOrder.emplace(&block, action->actionId);
                if (inserted) {
                    blockAnchor[&block] = canonical;
                } else if (action->actionId > order->second) {
                    order->second = action->actionId;
                    blockAnchor[&block] = canonical;
                }
            }
        }
    }
    if (!result.contract.actions.empty() && result.actionInstructions.empty())
        report_fatal_error(
            "GirFrameAnalysis: frame contract imported but no GirActionData survived lowering");

    struct IncomingValue {
        int value = 0;
        bool relative = false;

        bool operator==(const IncomingValue&) const = default;
    };
    using IncomingValues = std::map<int, IncomingValue>;
    std::unordered_map<BasicBlock*, std::map<uint64_t, IncomingValues>> incomingValues;
    std::unordered_map<uint64_t, std::vector<BasicBlock*>> anchorEntries;
    for (const auto& [anchor, blocks] : anchorBlocks) {
        std::vector<BasicBlock*>& entries = anchorEntries[anchor];
        for (BasicBlock* block : blocks) {
            const auto& predecessors = block->getPredecessors();
            const bool entersGroup =
                predecessors.empty() ||
                std::any_of(predecessors.begin(), predecessors.end(),
                            [&](BasicBlock* pred) { return !blocks.contains(pred); });
            if (entersGroup) entries.push_back(block);
        }
        // A self-contained loop can have no predecessor outside its own logical action group.
        if (entries.empty()) entries.assign(blocks.begin(), blocks.end());
    }
    // An ANCHOR is a logical identity: a block names one whether or not an instruction carries
    // that exact id, so an edge endpoint is valid if it is a realized action OR a named anchor.
    std::unordered_set<uint64_t> namedActions;
    for (const auto& [actionId, action] : result.contract.actions) {
        namedActions.insert(actionId);
        namedActions.insert(action.anchorAction);
    }
    for (const GirFrameIncomingSpec& incoming : result.contract.incomings) {
        if (!result.contract.generations.contains(incoming.genId))
            report_fatal_error("GirFrameAnalysis: INCOMING names unknown generation");
        if (!namedActions.contains(incoming.sourceAction) ||
            !namedActions.contains(incoming.destinationAction))
            report_fatal_error("GirFrameAnalysis: INCOMING names unknown action");
        auto destinations = anchorEntries.find(incoming.destinationAction);
        if (!realizedAnchors.contains(incoming.sourceAction) || destinations == anchorEntries.end())
            report_fatal_error(
                "GirFrameAnalysis: INCOMING action anchor has no physical realization");
        const int ring = result.contract.generations.at(incoming.genId).ring;
        int value = incoming.value % ring;
        if (value < 0) value += ring;
        for (BasicBlock* block : destinations->second) {
            auto [stored, inserted] = incomingValues[block][incoming.sourceAction].emplace(
                incoming.genId, IncomingValue{value, incoming.relative});
            if (!inserted && stored->second != IncomingValue{value, incoming.relative})
                report_fatal_error(
                    "GirFrameAnalysis: conflicting INCOMING values for physical block");
        }
    }

    const auto& loops = AM.getResult<LoopAnalysis>(function);
    std::unordered_map<int, const Loop*> genLoops;
    for (const auto& [genId, _gen] : result.contract.generations) {
        std::unordered_map<const Loop*, size_t> votes;
        for (const GirAccessSpec& access : result.contract.accesses) {
            if (access.genId != genId) continue;
            auto actionIt = result.actionInstructions.find(access.actionId);
            if (actionIt == result.actionInstructions.end()) continue;
            for (StinkyInstruction* inst : actionIt->second) {
                const Loop* loop = containingLoop(loops, instructionBlock.at(inst));
                if (loop) ++votes[loop];
            }
        }
        const Loop* chosen = nullptr;
        size_t best = 0;
        bool tied = false;
        for (const auto& [loop, count] : votes) {
            if (count > best) {
                chosen = loop;
                best = count;
                tied = false;
            } else if (count == best) {
                tied = true;
            }
        }
        if (chosen && !tied) {
            genLoops[genId] = chosen;
        } else if (!votes.empty()) {
            report_fatal_error("GirFrameAnalysis: generation maps ambiguously to ST loops");
        } else if (result.contract.generations.at(genId).ring > 1) {
            // Dropping it silently leaves its phase at 0 for the whole function, so every access
            // collapses onto `gdelta % ring` and distinct buffers alias onto one storage id.
            report_fatal_error("GirFrameAnalysis: rotating generation maps to no ST loop");
        }
    }

    for (const Loop& loop : loops) result.backEdges.insert(edgeKey(loop.latchBB, loop.headerBB));

    GirFrame base;
    for (const auto& [genId, _] : result.contract.generations) base.setPhase(genId, 0);
    BasicBlock* entry = function.getEntryBlock();
    if (!entry) return result;

    using FrameState = std::pair<GirFrame, uint64_t>;
    std::unordered_map<BasicBlock*, std::set<FrameState>> frameSets;
    std::deque<GirFrameNode> work;
    frameSets[entry].insert({base, 0});
    work.push_back({entry, base, 0});
    size_t stateCount = 1;
    constexpr size_t kMaxFrameStates = 1U << 20U;
    auto advanceNode = [&](const GirFrameNode& node, BasicBlock* successor) {
        GirFrame next =
            advanceFrame(node.frame, node.block, successor, loops, genLoops, result.contract);
        uint64_t outgoingAction = node.incomingAction;
        auto anchor = blockAnchor.find(node.block);
        if (anchor != blockAnchor.end()) outgoingAction = anchor->second;
        auto blockIncoming = incomingValues.find(successor);
        if (blockIncoming != incomingValues.end()) {
            auto source = blockIncoming->second.find(outgoingAction);
            if (source != blockIncoming->second.end())
                for (const auto& [genId, incoming] : source->second) {
                    int value = incoming.value;
                    if (incoming.relative) {
                        const int ring = result.contract.generations.at(genId).ring;
                        value = (node.frame.phaseOf(genId) + value) % ring;
                    }
                    next.setPhase(genId, value);
                }
        }
        const bool successorHasActions = blockActions.find(successor) != blockActions.end() &&
                                         !blockActions.at(successor).empty();
        return GirFrameNode{successor, std::move(next), successorHasActions ? 0 : outgoingAction};
    };
    while (!work.empty()) {
        GirFrameNode node = std::move(work.front());
        work.pop_front();
        for (BasicBlock* succ : node.block->getSuccessors()) {
            GirFrameNode next = advanceNode(node, succ);
            if (frameSets[succ].insert({next.frame, next.incomingAction}).second) {
                if (++stateCount > kMaxFrameStates)
                    report_fatal_error("GirFrameAnalysis: finite frame graph exceeds bound");
                work.push_back(std::move(next));
            }
        }
    }

    for (const auto& [block, states] : frameSets) {
        std::set<GirFrame> frames;
        for (const auto& [frame, _incomingAction] : states) frames.insert(frame);
        result.blockFrames[block] = std::vector<GirFrame>(frames.begin(), frames.end());
    }
    for (const auto& [block, states] : frameSets) {
        for (const auto& [frame, incomingAction] : states) {
            GirFrameNode node{block, frame, incomingAction};
            auto& successors = result.edges[node];
            for (BasicBlock* succ : block->getSuccessors())
                successors.push_back(advanceNode(node, succ));
        }
    }

    std::map<StorageKey, int> storageIds;
    for (size_t accessIndex = 0; accessIndex < result.contract.accesses.size(); ++accessIndex) {
        const GirAccessSpec& access = result.contract.accesses[accessIndex];
        auto actionSpec = result.contract.actions.find(access.actionId);
        auto instructions = result.actionInstructions.find(access.actionId);
        if (actionSpec == result.contract.actions.end() ||
            instructions == result.actionInstructions.end())
            continue;  // A folded GIR action can intentionally have no physical realization.
        for (StinkyInstruction* inst : instructions->second) {
            if (!instructionRealizes(actionSpec->second.kind, *inst)) continue;
            BasicBlock* block = instructionBlock.at(inst);
            std::set<GirFrame> frames;
            for (const auto& [frame, _incomingAction] : frameSets.at(block)) frames.insert(frame);
            for (const GirFrame& frame : frames) {
                int generation = access.absoluteGeneration;
                if (generation < 0) {
                    if (access.genId < 0)
                        report_fatal_error(
                            "GirFrameAnalysis: relative shared access has no generation");
                    generation = frame.phaseOf(access.genId) + access.gdelta;
                }
                generation %= access.ring;
                if (generation < 0) generation += access.ring;
                result.occurrences.push_back({inst, block, instructionIndex.at(inst), accessIndex,
                                              frame,
                                              concreteStorage(access, generation, storageIds)});
            }
        }
    }

    return result;
}

std::pair<StinkyInstruction*, GirFrame> lastBarrierBefore(const GirFrameAnalysis::Result& frames,
                                                          BasicBlock* block, const GirFrame& frame,
                                                          size_t limit) {
    const auto inBlock = [](BasicBlock* bb, size_t upTo) -> StinkyInstruction* {
        StinkyInstruction* found = nullptr;
        size_t index = 0;
        for (IRBase& node : *bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            if (index++ >= upTo) break;
            if (isFence(*inst) || isBarrier(*inst)) found = inst;
        }
        return found;
    };
    if (StinkyInstruction* here = inBlock(block, limit)) return {here, frame};

    using Key = std::pair<BasicBlock*, GirFrame>;
    std::map<Key, std::vector<Key>> preds;
    for (const auto& [from, tos] : frames.edges)
        for (const GirFrameNode& to : tos)
            preds[{to.block, to.frame}].push_back({from.block, from.frame});

    std::set<Key> seen{{block, frame}};
    std::deque<Key> work{{block, frame}};
    while (!work.empty()) {
        const Key node = work.front();
        work.pop_front();
        auto incoming = preds.find(node);
        if (incoming == preds.end()) continue;
        for (const Key& pred : incoming->second) {
            if (!seen.insert(pred).second) continue;
            if (StinkyInstruction* there =
                    inBlock(pred.first, std::numeric_limits<size_t>::max()))
                return {there, pred.second};
            work.push_back(pred);
        }
    }
    return {nullptr, frame};
}

GirFrameHazardAnalysis::Result GirFrameHazardAnalysis::run(Function& function,
                                                           AnalysisManager& AM) {
    Result result;
    const auto& frames = AM.getResult<GirFrameAnalysis>(function);
    if (frames.empty()) return result;

    OccurrenceRegistry registry;
    OccurrencesByNode occurrences;
    for (const GirAccessOccurrence& occurrence : frames.occurrences) {
        LiveTouchKey key{occurrence.frame, {occurrence.inst, occurrence.accessIndex}};
        if (!registry.emplace(key, &occurrence).second)
            report_fatal_error("GirFrameHazardAnalysis found a duplicate dynamic touch");
        occurrences[{occurrence.block, occurrence.frame}].push_back(&occurrence);
    }
    for (auto& [node, touches] : occurrences) {
        (void)node;
        std::sort(touches.begin(), touches.end(), [](const auto* a, const auto* b) {
            if (a->instructionIndex != b->instructionIndex)
                return a->instructionIndex < b->instructionIndex;
            if (a->inst != b->inst) return std::less<StinkyInstruction*>{}(a->inst, b->inst);
            return a->accessIndex < b->accessIndex;
        });
    }

    auto incoming = reachingHazards(frames, occurrences, registry);
    std::set<std::tuple<int, LiveTouchKey, LiveTouchKey, int>> emitted;
    const HazardEmitter emit = [&](const GirAccessOccurrence& producer,
                                   const GirAccessOccurrence& consumer, int gap) {
        const GirAccessSpec& producerSpec = frames.contract.accesses[producer.accessIndex];
        const GirAccessSpec& consumerSpec = frames.contract.accesses[consumer.accessIndex];
        GirHazardKind kind = GirHazardKind::WAR;
        if (producerSpec.isWrite)
            kind = consumerSpec.isWrite ? GirHazardKind::WAW : GirHazardKind::RAW;
        if (!producerSpec.isWrite && !consumerSpec.isWrite) return;
        if (kind == GirHazardKind::WAW &&
            wawOrderedByRead(producerSpec, consumerSpec, frames.contract))
            return;

        LiveTouchKey producerKey{producer.frame, {producer.inst, producer.accessIndex}};
        LiveTouchKey consumerKey{consumer.frame, {consumer.inst, consumer.accessIndex}};
        if (!emitted.emplace(static_cast<int>(kind), producerKey, consumerKey, gap).second) return;
        result.hazards.push_back({kind, producer.inst, consumer.inst, producer.block,
                                  consumer.block, producer.instructionIndex,
                                  consumer.instructionIndex, producer.frame, consumer.frame, gap,
                                  producerSpec.crossAgent || consumerSpec.crossAgent,
                                  actionOf(producer.inst), actionOf(consumer.inst)});
    };

    for (const auto& [node, state] : incoming) {
        auto touches = occurrences.find(node);
        static const std::vector<const GirAccessOccurrence*> empty;
        (void)walkHazards(node, touches == occurrences.end() ? empty : touches->second, state,
                          frames.contract, registry, &emit);
    }

    return result;
}

}  // namespace stinkytofu
