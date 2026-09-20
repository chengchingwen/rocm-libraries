/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"

#include <algorithm>
#include <charconv>
#include <deque>
#include <functional>
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

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> out;
    std::string item;
    std::istringstream in(text);
    while (std::getline(in, item, delimiter)) out.push_back(item);
    return out;
}

int hexDigit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::string percentDecode(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '%' && i + 2 < text.size()) {
            const int hi = hexDigit(text[i + 1]);
            const int lo = hexDigit(text[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(text[i]);
    }
    return out;
}

GirActionKind parseActionKind(const std::string& kind) {
    if (kind == "read") return GirActionKind::Read;
    if (kind == "copy") return GirActionKind::Copy;
    if (kind == "fence") return GirActionKind::Fence;
    if (kind == "wmma") return GirActionKind::Wmma;
    if (kind == "waitcnt") return GirActionKind::WaitCnt;
    return GirActionKind::Other;
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
        if (from == loop->latchBB)
            result.setPhase(genId, (result.phaseOf(genId) + gen.advance) % gen.ring);
        if (to == loop->headerBB && !(from == loop->latchBB && to == loop->headerBB))
            result.setPhase(genId, gen.entry % gen.ring);
    }
    return result;
}

bool intersects(const std::vector<int>& a, const std::vector<int>& b) {
    size_t i = 0;
    size_t j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) return true;
        if (a[i] < b[j])
            ++i;
        else
            ++j;
    }
    return false;
}

bool disjointRegions(const GirAccessSpec& a, const GirAccessSpec& b) {
    if (a.operand != b.operand) return true;
    const size_t axes = std::min(a.regions.size(), b.regions.size());
    for (size_t axis = 0; axis < axes; ++axis)
        if (!intersects(a.regions[axis], b.regions[axis])) return true;
    return false;
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

std::vector<std::vector<int>> regionCombinations(const GirAccessSpec& access) {
    std::vector<std::vector<int>> combinations(1);
    for (const std::vector<int>& axis : access.regions) {
        std::vector<std::vector<int>> next;
        for (const std::vector<int>& prefix : combinations)
            for (int value : axis) {
                std::vector<int> item = prefix;
                item.push_back(value);
                next.push_back(std::move(item));
            }
        combinations = std::move(next);
    }
    return combinations;
}

using StorageKey = std::tuple<std::string, std::vector<int>, int>;

std::vector<int> concreteStorage(const GirAccessSpec& access, int generation,
                                 std::map<StorageKey, int>& storageIds) {
    std::vector<int> result;
    for (std::vector<int>& region : regionCombinations(access)) {
        StorageKey key{access.operand, std::move(region), generation};
        auto [it, inserted] =
            storageIds.emplace(std::move(key), static_cast<int>(storageIds.size()));
        (void)inserted;
        result.push_back(it->second);
    }
    std::sort(result.begin(), result.end());
    return result;
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
        for (int storage : touch->storage) {
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
    std::string line;
    if (!std::getline(input, line) || line != "GIR_FRAME_CONTRACT_V1")
        report_fatal_error("GIR frame contract: missing GIR_FRAME_CONTRACT_V1 header");
    result.version = 1;

    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::istringstream fields(line);
        std::string tag;
        fields >> tag;
        if (tag == "GEN") {
            std::string id, ring, entry, advance;
            if (!(fields >> id >> ring >> entry >> advance))
                report_fatal_error("GIR frame contract: malformed GEN line");
            GirGenerationSpec gen{parseInt(id, "gen id"), parseInt(ring, "ring"),
                                  parseInt(entry, "entry"), parseInt(advance, "advance")};
            if (gen.ring < 1) report_fatal_error("GIR frame contract: ring must be positive");
            result.generations[gen.id] = gen;
        } else if (tag == "ACTION") {
            std::string id, kind;
            if (!(fields >> id >> kind))
                report_fatal_error("GIR frame contract: malformed ACTION line");
            GirActionSpec action;
            action.id = parseU64(id, "action id");
            action.kind = parseActionKind(kind);
            result.actions[action.id] = action;
        } else if (tag == "ACCESS") {
            std::string action, write, gen, ring, delta, absolute, cross, operand, regions;
            if (!(fields >> action >> write >> gen >> ring >> delta >> absolute >> cross >>
                  operand >> regions))
                report_fatal_error("GIR frame contract: malformed ACCESS line");
            GirAccessSpec access;
            access.actionId = parseU64(action, "access action");
            access.isWrite = parseInt(write, "write") != 0;
            access.genId = parseInt(gen, "access gen");
            access.ring = parseInt(ring, "access ring");
            access.gdelta = parseInt(delta, "gdelta");
            access.absoluteGeneration = parseInt(absolute, "absolute generation");
            access.crossAgent = parseInt(cross, "cross-agent") != 0;
            access.operand = percentDecode(operand);
            if (regions != "-") {
                for (const std::string& axis : split(regions, '/')) {
                    std::vector<int> values;
                    for (const std::string& value : split(axis, ','))
                        values.push_back(parseInt(value, "region value"));
                    std::sort(values.begin(), values.end());
                    if (values.empty()) report_fatal_error("GIR frame contract: empty region axis");
                    access.regions.push_back(std::move(values));
                }
            }
            if (access.ring < 1)
                report_fatal_error("GIR frame contract: ACCESS ring must be positive");
            const size_t index = result.accesses.size();
            result.accesses.push_back(std::move(access));
            auto actionIt = result.actions.find(result.accesses.back().actionId);
            if (actionIt == result.actions.end())
                report_fatal_error("GIR frame contract: ACCESS names unknown action");
            actionIt->second.accesses.push_back(index);
        } else if (tag == "REL") {
            std::string fence, kind, producer, consumer, gap;
            if (!(fields >> fence >> kind >> producer >> consumer >> gap))
                report_fatal_error("GIR frame contract: malformed REL line");
            result.relations.push_back({parseU64(fence, "fence action"), parseHazardKind(kind),
                                        parseU64(producer, "producer action"),
                                        parseU64(consumer, "consumer action"),
                                        parseInt(gap, "gap")});
        } else {
            report_fatal_error("GIR frame contract: unknown record '" + tag + "'");
        }
    }
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
    return result ^ (GirFrameHash{}(node.frame) + 0x9e3779b9U + (result << 6U) + (result >> 2U));
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
    for (BasicBlock& block : function) {
        size_t index = 0;
        for (IRBase& node : block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            instructionIndex[inst] = index++;
            instructionBlock[inst] = &block;
            if (const auto* action = inst->getModifier<GirActionData>()) {
                result.actionInstructions[action->actionId].push_back(inst);
            }
        }
    }
    if (!result.contract.actions.empty() && result.actionInstructions.empty())
        report_fatal_error(
            "GirFrameAnalysis: frame contract imported but no GirActionData survived lowering");

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
        if (chosen && !tied)
            genLoops[genId] = chosen;
        else if (!votes.empty())
            report_fatal_error("GirFrameAnalysis: generation maps ambiguously to ST loops");
    }

    for (const Loop& loop : loops) result.backEdges.insert(edgeKey(loop.latchBB, loop.headerBB));

    GirFrame base;
    for (const auto& [genId, _] : result.contract.generations) base.setPhase(genId, 0);
    BasicBlock* entry = function.getEntryBlock();
    if (!entry) return result;

    std::unordered_map<BasicBlock*, std::set<GirFrame>> frameSets;
    std::deque<GirFrameNode> work;
    frameSets[entry].insert(base);
    work.push_back({entry, base});
    size_t stateCount = 1;
    constexpr size_t kMaxFrameStates = 1U << 20U;
    while (!work.empty()) {
        GirFrameNode node = std::move(work.front());
        work.pop_front();
        for (BasicBlock* succ : node.block->getSuccessors()) {
            GirFrame next =
                advanceFrame(node.frame, node.block, succ, loops, genLoops, result.contract);
            if (frameSets[succ].insert(next).second) {
                if (++stateCount > kMaxFrameStates)
                    report_fatal_error("GirFrameAnalysis: finite frame graph exceeds bound");
                work.push_back({succ, std::move(next)});
            }
        }
    }

    for (auto& [block, frames] : frameSets)
        result.blockFrames[block] = std::vector<GirFrame>(frames.begin(), frames.end());
    for (const auto& [block, frames] : result.blockFrames) {
        for (const GirFrame& frame : frames) {
            GirFrameNode node{block, frame};
            auto& successors = result.edges[node];
            for (BasicBlock* succ : block->getSuccessors())
                successors.push_back(
                    {succ, advanceFrame(frame, block, succ, loops, genLoops, result.contract)});
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
            for (const GirFrame& frame : result.frames(block)) {
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
