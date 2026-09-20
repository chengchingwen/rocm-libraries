/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "stinkytofu/transforms/asm/waitcnt/GirFrameCounterFlow.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"

namespace stinkytofu::waitcnt {
namespace {

constexpr size_t kTrackedCounterCount = 2;

size_t trackedIndex(CounterKind counter) {
    if (counter == CK_DS) return 0;
    if (counter == CK_Tensor) return 1;
    report_fatal_error("GIR frame counter flow received an untracked counter");
}

CounterKind counterAt(size_t index) {
    return index == 0 ? CK_DS : CK_Tensor;
}

struct IssueKey {
    StinkyInstruction* inst = nullptr;
    GirFrame frame;

    bool operator==(const IssueKey&) const = default;
    bool operator<(const IssueKey& other) const {
        if (inst != other.inst) return std::less<StinkyInstruction*>{}(inst, other.inst);
        return frame < other.frame;
    }
};

struct CounterQueue {
    std::vector<IssueKey> ops;

    bool operator==(const CounterQueue&) const = default;
    bool operator<(const CounterQueue& other) const {
        return std::lexicographical_compare(ops.begin(), ops.end(), other.ops.begin(),
                                            other.ops.end());
    }

    void append(IssueKey issue) {
        // `(instruction, frame)` is a complete identity in the finite ring quotient. A repeated
        // occurrence replaces its prior queue copy instead of manufacturing an unbounded trip id.
        ops.erase(std::remove(ops.begin(), ops.end(), issue), ops.end());
        ops.push_back(std::move(issue));
    }

    int countFrom(const IssueKey& issue) const {
        auto found = std::find(ops.begin(), ops.end(), issue);
        if (found != ops.end()) return static_cast<int>(std::distance(found, ops.end()));

        // ST's reconstructed CFG can merge scaffold paths at a different frame cut than GIR while
        // still carrying the same physical producer instruction. The frame hazard is authoritative:
        // if that producer is in flight under another reachable frame, use its oldest occurrence
        // (the strictest rank) rather than silently dropping the wait for lack of an exact label.
        int strictest = 0;
        for (auto it = ops.begin(); it != ops.end(); ++it)
            if (it->inst == issue.inst)
                strictest = std::max(strictest, static_cast<int>(std::distance(it, ops.end())));
        return strictest;
    }

    void applyWait(int keep) {
        if (keep <= 0) {
            ops.clear();
        } else if (static_cast<int>(ops.size()) > keep) {
            ops.erase(ops.begin(), ops.end() - keep);
        }
    }
};

struct FlowState {
    GirFrameNode node;
    BasicBlock* incomingPred = nullptr;
    std::set<int> trips;
    std::array<CounterQueue, kTrackedCounterCount> queues;

    bool operator<(const FlowState& other) const {
        if (node.block != other.node.block)
            return std::less<BasicBlock*>{}(node.block, other.node.block);
        if (!(node.frame == other.node.frame)) return node.frame < other.node.frame;
        if (incomingPred != other.incomingPred)
            return std::less<BasicBlock*>{}(incomingPred, other.incomingPred);
        if (trips != other.trips)
            return std::lexicographical_compare(trips.begin(), trips.end(), other.trips.begin(),
                                                other.trips.end());
        return std::lexicographical_compare(queues.begin(), queues.end(), other.queues.begin(),
                                            other.queues.end());
    }
};

struct StaticSite {
    StinkyInstruction* anchor = nullptr;
    CounterKind counter = CK_Count;

    bool operator==(const StaticSite&) const = default;
    bool operator<(const StaticSite& other) const {
        if (anchor != other.anchor) return std::less<StinkyInstruction*>{}(anchor, other.anchor);
        return counter < other.counter;
    }
};

struct TailSite {
    BasicBlock* block = nullptr;
    CounterKind counter = CK_Count;

    bool operator==(const TailSite&) const = default;
    bool operator<(const TailSite& other) const {
        if (block != other.block) return std::less<BasicBlock*>{}(block, other.block);
        return counter < other.counter;
    }
};

struct DynamicSite {
    StinkyInstruction* anchor = nullptr;
    GirFrame frame;
    CounterKind counter = CK_Count;

    bool operator<(const DynamicSite& other) const {
        if (anchor != other.anchor) return std::less<StinkyInstruction*>{}(anchor, other.anchor);
        if (!(frame == other.frame)) return frame < other.frame;
        return counter < other.counter;
    }
};

struct Potential {
    IssueKey producer;
};

struct RequirementSummary {
    int strictest = WaitCountSpec::kUnused;
    std::map<BasicBlock*, int, std::less<BasicBlock*>> byPred;
    std::set<IssueKey> producers;
    std::map<BasicBlock*, std::set<IssueKey>, std::less<BasicBlock*>> producersByPred;

    void record(BasicBlock* pred, int wait, const IssueKey& producer) {
        if (wait < 0) return;
        if (strictest == WaitCountSpec::kUnused || wait < strictest) strictest = wait;
        auto [it, inserted] = byPred.emplace(pred, wait);
        if (!inserted) it->second = std::min(it->second, wait);
        producers.insert(producer);
        producersByPred[pred].insert(producer);
    }
};

using DecisionMap = std::map<StaticSite, int>;
using TailDecisionMap = std::map<TailSite, int>;
using TailProducerMap = std::map<TailSite, std::set<IssueKey>>;
using PotentialMap = std::map<DynamicSite, std::vector<Potential>>;
using RequirementMap = std::map<StaticSite, RequirementSummary>;
using FeasibleDomainMap = std::unordered_map<GirFrameNode, std::set<int>, GirFrameNodeHash>;

CounterKind counterFor(const GirFrameHazard& hazard) {
    if (hazard.kind == GirHazardKind::RAW) return classifyMemOp(*hazard.producer);
    if (classifyMemOp(*hazard.producer) == CK_DS) return CK_DS;
    if (classifyMemOp(*hazard.consumer) == CK_DS) return CK_DS;
    return CK_Tensor;
}

StinkyInstruction* fenceForAction(const GirFrameAnalysis::Result& frames, uint64_t action) {
    auto found = frames.actionInstructions.find(action);
    if (found == frames.actionInstructions.end()) return nullptr;
    for (StinkyInstruction* inst : found->second)
        if (isFence(*inst) || isBarrier(*inst)) return inst;
    return nullptr;
}

size_t instructionIndex(const StinkyInstruction* target) {
    size_t index = 0;
    for (const IRBase& node : *target->getParent()) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (!inst) continue;
        if (inst == target) return index;
        ++index;
    }
    report_fatal_error("GIR frame fence is detached from its ST basic block");
}

bool positiveNodeReach(const GirFrameAnalysis::Result& frames, const GirFrameNode& from,
                       const GirFrameNode& to) {
    std::deque<GirFrameNode> work;
    std::unordered_set<GirFrameNode, GirFrameNodeHash> seen;
    auto start = frames.edges.find(from);
    if (start == frames.edges.end()) return false;
    work.insert(work.end(), start->second.begin(), start->second.end());
    while (!work.empty()) {
        GirFrameNode node = std::move(work.front());
        work.pop_front();
        if (node == to) return true;
        if (!seen.insert(node).second) continue;
        auto next = frames.edges.find(node);
        if (next != frames.edges.end())
            work.insert(work.end(), next->second.begin(), next->second.end());
    }
    return false;
}

bool occurrencePrecedes(const GirFrameAnalysis::Result& frames, BasicBlock* fromBlock,
                        const GirFrame& fromFrame, size_t fromIndex, BasicBlock* toBlock,
                        const GirFrame& toFrame, size_t toIndex) {
    GirFrameNode from{fromBlock, fromFrame};
    GirFrameNode to{toBlock, toFrame};
    if (from == to && fromIndex < toIndex) return true;
    return positiveNodeReach(frames, from, to);
}

StinkyInstruction* reconstructedFence(const GirFrameHazard& hazard,
                                      const GirFrameAnalysis::Result& frames) {
    for (const auto& [actionId, action] : frames.contract.actions) {
        if (action.kind != GirActionKind::Fence) continue;
        auto physical = frames.actionInstructions.find(actionId);
        if (physical == frames.actionInstructions.end()) continue;
        for (StinkyInstruction* candidate : physical->second) {
            if (!isFence(*candidate) && !isBarrier(*candidate)) continue;
            BasicBlock* fenceBlock = candidate->getParent();
            const size_t fenceIndex = instructionIndex(candidate);
            for (const GirFrame& fenceFrame : frames.frames(fenceBlock))
                if (occurrencePrecedes(frames, hazard.producerBlock, hazard.producerFrame,
                                       hazard.producerIndex, fenceBlock, fenceFrame, fenceIndex) &&
                    occurrencePrecedes(frames, fenceBlock, fenceFrame, fenceIndex,
                                       hazard.consumerBlock, hazard.consumerFrame,
                                       hazard.consumerIndex))
                    return candidate;
        }
    }
    return nullptr;
}

StinkyInstruction* relationFence(const GirFrameHazard& hazard,
                                 const GirFrameAnalysis::Result& frames) {
    StinkyInstruction* result = nullptr;
    for (const GirFenceRelationSpec& relation : frames.contract.relations) {
        if (relation.kind != hazard.kind || relation.producerAction != hazard.producerAction ||
            relation.consumerAction != hazard.consumerAction ||
            (relation.gap == 0) != (hazard.gap == 0))
            continue;
        StinkyInstruction* candidate = fenceForAction(frames, relation.fenceAction);
        if (!candidate)
            report_fatal_error("GIR relation-owning fence has no physical or virtual ST fence");
        if (result && result != candidate)
            report_fatal_error("GIR frame hazard has several relation-owning ST fences");
        result = candidate;
    }
    if (!result) result = reconstructedFence(hazard, frames);
    if (!result) {
        std::ostringstream message;
        message << "Cross-agent ST frame hazard has no exact GIR relation-owning fence"
                << " kind=" << static_cast<int>(hazard.kind)
                << " producerAction=" << hazard.producerAction
                << " consumerAction=" << hazard.consumerAction << " gap=" << hazard.gap
                << " producerBlock=" << hazard.producerBlock->getLabel()
                << " consumerBlock=" << hazard.consumerBlock->getLabel();
        report_fatal_error(message.str());
    }
    return result;
}

FeasibleDomainMap computeFeasibleDomains(Function& function,
                                         const GirFrameAnalysis::Result& frames);
bool hazardFeasible(Function& function, const GirFrameAnalysis::Result& frames,
                    const FeasibleDomainMap& feasible, const GirFrameHazard& hazard);

PotentialMap buildPotentials(Function& function, const GirFrameAnalysis::Result& frames,
                             const GirFrameHazardAnalysis::Result& hazards) {
    PotentialMap result;
    const FeasibleDomainMap feasible = computeFeasibleDomains(function, frames);
    std::set<std::tuple<StinkyInstruction*, GirFrame, CounterKind, StinkyInstruction*, GirFrame>>
        seen;
    for (const GirFrameHazard& hazard : hazards.hazards) {
        if (!hazardFeasible(function, frames, feasible, hazard)) continue;
        CounterKind counter = counterFor(hazard);
        if (counter != CK_DS && counter != CK_Tensor) continue;
        StinkyInstruction* anchor =
            hazard.crossAgent ? relationFence(hazard, frames) : hazard.consumer;
        const GirFrame anchorFrame =
            hazard.crossAgent && anchor->getParent() == hazard.producerBlock ? hazard.producerFrame
                                                                             : hazard.consumerFrame;
        const auto& availableFrames = frames.frames(anchor->getParent());
        if (std::find(availableFrames.begin(), availableFrames.end(), anchorFrame) ==
            availableFrames.end())
            report_fatal_error(
                "GIR relation fence frame is absent from the reconstructed ST block");
        auto key =
            std::make_tuple(anchor, anchorFrame, counter, hazard.producer, hazard.producerFrame);
        if (!seen.insert(key).second) continue;
        result[{anchor, anchorFrame, counter}].push_back({{hazard.producer, hazard.producerFrame}});
    }
    return result;
}

int observedWait(const StinkyInstruction& inst, CounterKind counter) {
    if (counter == CK_DS) {
        if (const auto* wait = inst.getModifier<SWaitCntData>()) return wait->dlcnt;
    } else if (counter == CK_Tensor) {
        if (const auto* wait = inst.getModifier<SWaitTensorCntData>()) return wait->tlcnt;
    }
    return WaitCountSpec::kUnused;
}

bool isTripCompare(const StinkyInstruction& inst) {
    const auto opcode = inst.getUnifiedOpcode();
    return opcode == GFX::s_cmp_eq_u32 || opcode == GFX::s_cmp_eq_i32 ||
           opcode == GFX::s_cmp_lg_u32 || opcode == GFX::s_cmp_lg_i32 ||
           opcode == GFX::s_cmp_lt_u32 || opcode == GFX::s_cmp_lt_i32 ||
           opcode == GFX::s_cmp_le_u32 || opcode == GFX::s_cmp_le_i32 ||
           opcode == GFX::s_cmp_gt_u32 || opcode == GFX::s_cmp_gt_i32 ||
           opcode == GFX::s_cmp_ge_u32 || opcode == GFX::s_cmp_ge_i32;
}

const StinkyInstruction* definingTripCompare(const StinkyInstruction& branch) {
    for (StinkyInstruction* source : branch.getSources())
        if (source && isTripCompare(*source)) return source;

    const BasicBlock* block = branch.getParent();
    for (auto it = block->rbegin(); it != block->rend(); ++it) {
        auto* candidate = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (candidate && candidate != &branch && isTripCompare(*candidate)) return candidate;
    }
    return nullptr;
}

std::optional<int> compareLiteral(const StinkyInstruction& compare) {
    for (const StinkyRegister& operand : compare.getSrcRegs()) {
        if (operand.dataType == StinkyRegister::Type::LiteralInt) return operand.getLiteralInt();
        if (operand.dataType == StinkyRegister::Type::LiteralString) {
            const std::string text = operand.getLiteralString();
            char* end = nullptr;
            const long value = std::strtol(text.c_str(), &end, 0);
            if (end == text.c_str() + text.size()) return static_cast<int>(value);
        }
    }
    return std::nullopt;
}

bool isTripControlTarget(const std::string& label) {
    return label.find("skipPGR") != std::string::npos || label.find("toPGR") != std::string::npos ||
           label.find("LoopEndL") != std::string::npos ||
           label.find("NoGlobalLoadLoop") != std::string::npos;
}

std::optional<bool> evaluateTripCompare(const StinkyInstruction& compare, int trip) {
    const std::optional<int> literal = compareLiteral(compare);
    if (!literal) return std::nullopt;
    const auto opcode = compare.getUnifiedOpcode();
    if (opcode == GFX::s_cmp_eq_u32 || opcode == GFX::s_cmp_eq_i32) return trip == *literal;
    if (opcode == GFX::s_cmp_lg_u32 || opcode == GFX::s_cmp_lg_i32) return trip != *literal;
    if (opcode == GFX::s_cmp_lt_u32 || opcode == GFX::s_cmp_lt_i32) return trip < *literal;
    if (opcode == GFX::s_cmp_le_u32 || opcode == GFX::s_cmp_le_i32) return trip <= *literal;
    if (opcode == GFX::s_cmp_gt_u32 || opcode == GFX::s_cmp_gt_i32) return trip > *literal;
    if (opcode == GFX::s_cmp_ge_u32 || opcode == GFX::s_cmp_ge_i32) return trip >= *literal;
    return std::nullopt;
}

std::set<int> edgeTripDomain(BasicBlock* block, BasicBlock* successor,
                             const std::set<int>& domain) {
    IRBase* terminator = block->getTerminator();
    auto* branch = terminator ? dyn_cast<StinkyInstruction>(terminator) : nullptr;
    if (!branch || !isConditionalBranch(*branch)) return domain;
    const std::vector<std::string> targets = getBranchTargets(*branch);
    if (targets.size() != 1 || !isTripControlTarget(targets.front())) return domain;
    const StinkyInstruction* compare = definingTripCompare(*branch);
    if (!compare) return domain;

    const bool taken = successor->getLabel() == targets.front();
    bool branchOnTrue = false;
    if (branch->getUnifiedOpcode() == GFX::s_cbranch_scc1)
        branchOnTrue = true;
    else if (branch->getUnifiedOpcode() != GFX::s_cbranch_scc0)
        return domain;

    std::set<int> allowed;
    for (int trip : domain) {
        const std::optional<bool> comparison = evaluateTripCompare(*compare, trip);
        if (!comparison) return domain;
        const bool takesBranch = *comparison == branchOnTrue;
        if (takesBranch == taken) allowed.insert(trip);
    }
    return allowed;
}

std::set<int> initialTripDomain(Function& function) {
    int limit = 4;
    for (BasicBlock& block : function) {
        IRBase* terminator = block.getTerminator();
        auto* branch = terminator ? dyn_cast<StinkyInstruction>(terminator) : nullptr;
        if (!branch || !isConditionalBranch(*branch)) continue;
        const std::vector<std::string> targets = getBranchTargets(*branch);
        if (targets.size() != 1 || !isTripControlTarget(targets.front())) continue;
        const StinkyInstruction* compare = definingTripCompare(*branch);
        if (!compare) continue;
        const std::optional<int> literal = compareLiteral(*compare);
        if (literal) limit = std::max(limit, *literal + 3);
    }
    std::set<int> result;
    for (int trip = 1; trip <= limit; ++trip) result.insert(trip);
    return result;
}

FeasibleDomainMap computeFeasibleDomains(Function& function,
                                         const GirFrameAnalysis::Result& frames) {
    FeasibleDomainMap feasible;
    std::deque<GirFrameNode> work;
    BasicBlock* entry = function.getEntryBlock();
    if (!entry) return feasible;
    const std::set<int> initial = initialTripDomain(function);
    for (const GirFrame& frame : frames.frames(entry)) {
        GirFrameNode node{entry, frame};
        feasible[node] = initial;
        work.push_back(node);
    }

    while (!work.empty()) {
        GirFrameNode node = std::move(work.front());
        work.pop_front();
        auto successors = frames.edges.find(node);
        if (successors == frames.edges.end()) continue;
        for (const GirFrameNode& successor : successors->second) {
            const std::set<int> allowed =
                edgeTripDomain(node.block, successor.block, feasible[node]);
            if (allowed.empty()) continue;
            std::set<int>& destination = feasible[successor];
            const size_t before = destination.size();
            destination.insert(allowed.begin(), allowed.end());
            if (destination.size() != before) work.push_back(successor);
        }
    }
    return feasible;
}

bool domainsIntersect(const std::set<int>& lhs, const std::set<int>& rhs) {
    auto a = lhs.begin();
    auto b = rhs.begin();
    while (a != lhs.end() && b != rhs.end()) {
        if (*a == *b) return true;
        if (*a < *b)
            ++a;
        else
            ++b;
    }
    return false;
}

bool hazardFeasible(Function& function, const GirFrameAnalysis::Result& frames,
                    const FeasibleDomainMap& feasible, const GirFrameHazard& hazard) {
    (void)function;
    const GirFrameNode source{hazard.producerBlock, hazard.producerFrame};
    const GirFrameNode target{hazard.consumerBlock, hazard.consumerFrame};
    auto sourceDomain = feasible.find(source);
    auto targetDomain = feasible.find(target);
    if (sourceDomain == feasible.end() || targetDomain == feasible.end()) return false;
    if (source == target && hazard.producerIndex < hazard.consumerIndex)
        return domainsIntersect(sourceDomain->second, targetDomain->second);

    std::deque<std::pair<GirFrameNode, std::set<int>>> work;
    auto first = frames.edges.find(source);
    if (first == frames.edges.end()) return false;
    for (const GirFrameNode& successor : first->second) {
        std::set<int> domain = edgeTripDomain(source.block, successor.block, sourceDomain->second);
        if (!domain.empty()) work.push_back({successor, std::move(domain)});
    }

    FeasibleDomainMap seen;
    while (!work.empty()) {
        auto [node, domain] = std::move(work.front());
        work.pop_front();
        std::set<int> fresh;
        std::set<int>& prior = seen[node];
        std::set_difference(domain.begin(), domain.end(), prior.begin(), prior.end(),
                            std::inserter(fresh, fresh.end()));
        if (fresh.empty()) continue;
        prior.insert(fresh.begin(), fresh.end());
        if (node == target && domainsIntersect(fresh, targetDomain->second)) return true;
        auto successors = frames.edges.find(node);
        if (successors == frames.edges.end()) continue;
        for (const GirFrameNode& successor : successors->second) {
            std::set<int> next = edgeTripDomain(node.block, successor.block, fresh);
            if (!next.empty()) work.push_back({successor, std::move(next)});
        }
    }
    return false;
}

void applyDecision(const DecisionMap& decisions, StinkyInstruction* anchor,
                   std::array<CounterQueue, kTrackedCounterCount>& queues) {
    for (size_t i = 0; i < kTrackedCounterCount; ++i) {
        auto found = decisions.find({anchor, counterAt(i)});
        if (found != decisions.end()) queues[i].applyWait(found->second);
    }
}

RequirementMap simulate(Function& function, const GirFrameAnalysis::Result& frames,
                        const PotentialMap& potentials, const DecisionMap& decisions,
                        const TailDecisionMap& tailDecisions) {
    RequirementMap requirements;
    std::deque<FlowState> work;
    std::set<FlowState> visited;

    BasicBlock* entry = function.getEntryBlock();
    if (!entry) return requirements;
    const std::set<int> initialDomain = initialTripDomain(function);
    for (const GirFrame& frame : frames.frames(entry))
        work.push_back({{entry, frame}, nullptr, initialDomain, {}});

    while (!work.empty()) {
        FlowState state = std::move(work.front());
        work.pop_front();
        if (!visited.insert(state).second) continue;

        for (IRBase& node : *state.node.block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;

            for (size_t i = 0; i < kTrackedCounterCount; ++i) {
                CounterKind counter = counterAt(i);
                DynamicSite dynamic{inst, state.node.frame, counter};
                auto found = potentials.find(dynamic);
                if (found == potentials.end()) continue;
                StaticSite site{inst, counter};
                for (const Potential& potential : found->second) {
                    const int count = state.queues[i].countFrom(potential.producer);
                    if (count > 0)
                        requirements[site].record(state.incomingPred, waitToDrain(counter, count),
                                                  potential.producer);
                }
            }

            applyDecision(decisions, inst, state.queues);
            for (size_t i = 0; i < kTrackedCounterCount; ++i) {
                const int wait = observedWait(*inst, counterAt(i));
                if (wait >= 0) state.queues[i].applyWait(wait);
            }

            CounterKind issued = classifyMemOp(*inst);
            if (issued == CK_DS || issued == CK_Tensor)
                state.queues[trackedIndex(issued)].append({inst, state.node.frame});
        }

        for (size_t i = 0; i < kTrackedCounterCount; ++i) {
            auto tail = tailDecisions.find({state.node.block, counterAt(i)});
            if (tail != tailDecisions.end()) state.queues[i].applyWait(tail->second);
        }

        auto successors = frames.edges.find(state.node);
        if (successors == frames.edges.end()) continue;
        for (const GirFrameNode& successor : successors->second) {
            std::set<int> domain = edgeTripDomain(state.node.block, successor.block, state.trips);
            if (domain.empty()) continue;
            FlowState next = state;
            next.node = successor;
            next.incomingPred = state.node.block;
            next.trips = std::move(domain);
            work.push_back(std::move(next));
        }
    }
    return requirements;
}

int& counterField(WaitCountSpec& spec, CounterKind counter) {
    if (counter == CK_DS) return spec.dsCount;
    if (counter == CK_Tensor) return spec.tensorCount;
    report_fatal_error("GIR frame wait plan asked for an unsupported counter field");
}

void tightenField(WaitCountSpec& spec, CounterKind counter, int wait) {
    int& field = counterField(spec, counter);
    if (field == WaitCountSpec::kUnused || wait < field) field = wait;
}

void attachTensorTokens(WaitCountSpec& spec, const std::set<IssueKey>& producers) {
    if (spec.tensorCount == WaitCountSpec::kUnused) return;
    for (const IssueKey& producer : producers) {
        const auto* tokens = producer.inst->getModifier<MemTokenData>();
        if (tokens)
            spec.tensorTokens.insert(spec.tensorTokens.end(), tokens->tokens.begin(),
                                     tokens->tokens.end());
    }
    std::sort(spec.tensorTokens.begin(), spec.tensorTokens.end());
    spec.tensorTokens.erase(std::unique(spec.tensorTokens.begin(), spec.tensorTokens.end()),
                            spec.tensorTokens.end());
}

void attachTensorTokens(WaitCountSpec& spec, const RequirementSummary& summary) {
    attachTensorTokens(spec, summary.producers);
}

std::pair<int, bool> prefixIssuesBefore(StinkyInstruction* anchor, CounterKind counter) {
    int issues = 0;
    for (IRBase& node : *anchor->getParent()) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (!inst) continue;
        if (inst == anchor) break;
        if (observedWait(*inst, counter) >= 0) return {issues, true};
        if (classifyMemOp(*inst) == counter) ++issues;
    }
    return {issues, false};
}

struct PromotionPlan {
    DecisionMap anchors;
    TailDecisionMap tails;
    TailProducerMap tailProducers;
};

PromotionPlan derivePromotions(const RequirementMap& requirements, const DecisionMap& decisions) {
    PromotionPlan result;
    for (const auto& [site, summary] : requirements) {
        auto current = decisions.find(site);
        if (current == decisions.end() || summary.strictest == WaitCountSpec::kUnused) continue;
        BasicBlock* block = site.anchor->getParent();
        const auto& predecessors = block->getPredecessors();
        size_t constrainedPredecessors = 0;
        for (const auto& [pred, wait] : summary.byPred) {
            (void)wait;
            if (pred) ++constrainedPredecessors;
        }
        if (predecessors.size() <= 1 || constrainedPredecessors < 2) continue;

        const auto [prefixIssues, prefixHasWait] = prefixIssuesBefore(site.anchor, site.counter);
        if (prefixHasWait) continue;

        int common = WaitCountSpec::kUnused;
        for (const auto& [pred, wait] : summary.byPred)
            if (pred) common = std::max(common, wait);
        if (common <= current->second) continue;

        PromotionPlan local;
        bool valid = true;
        for (const auto& [pred, need] : summary.byPred) {
            if (!pred || need >= common) continue;
            if (pred->getSuccessors().size() != 1 || pred->getSuccessors().front() != block) {
                valid = false;
                break;
            }
            const int edgeWait = need - prefixIssues;
            if (edgeWait < 0) {
                valid = false;
                break;
            }
            TailSite edge{pred, site.counter};
            auto [tail, inserted] = local.tails.emplace(edge, edgeWait);
            if (!inserted) tail->second = std::min(tail->second, edgeWait);
            auto producers = summary.producersByPred.find(pred);
            if (producers != summary.producersByPred.end())
                local.tailProducers[edge].insert(producers->second.begin(),
                                                 producers->second.end());
        }
        if (!valid || local.tails.empty()) continue;
        result.anchors[site] = common;
        for (const auto& [edge, wait] : local.tails) {
            auto [tail, inserted] = result.tails.emplace(edge, wait);
            if (!inserted) tail->second = std::min(tail->second, wait);
        }
        for (const auto& [edge, producers] : local.tailProducers)
            result.tailProducers[edge].insert(producers.begin(), producers.end());
    }
    return result;
}

RequirementMap closeDecisions(Function& function, const GirFrameAnalysis::Result& frames,
                              const PotentialMap& potentials, DecisionMap& decisions,
                              const TailDecisionMap& tailDecisions) {
    const size_t rounds = frames.edges.size() + 1;
    for (size_t round = 0; round < rounds; ++round) {
        RequirementMap requirements =
            simulate(function, frames, potentials, decisions, tailDecisions);
        DecisionMap next;
        for (const auto& [site, summary] : requirements)
            if (summary.strictest != WaitCountSpec::kUnused) next[site] = summary.strictest;
        if (next == decisions) return requirements;
        decisions = std::move(next);
    }
    report_fatal_error("GIR finite-frame counter flow did not close at its frame-node bound");
}

WaitInsertionPlan materializePlan(const DecisionMap& decisions,
                                  const TailDecisionMap& tailDecisions,
                                  const TailProducerMap& tailProducers,
                                  const RequirementMap& requirements) {
    WaitInsertionPlan plan;
    for (const auto& [site, wait] : decisions) {
        WaitCountSpec& spec = plan.anchorWaits[site.anchor];
        tightenField(spec, site.counter, wait);
        auto summary = requirements.find(site);
        if (site.counter == CK_Tensor && summary != requirements.end())
            attachTensorTokens(spec, summary->second);
    }

    std::map<BasicBlock*, WaitCountSpec, std::less<BasicBlock*>> tailSpecs;
    for (const auto& [site, wait] : tailDecisions) {
        WaitCountSpec& spec = tailSpecs[site.block];
        tightenField(spec, site.counter, wait);
        auto producers = tailProducers.find(site);
        if (site.counter == CK_Tensor && producers != tailProducers.end())
            attachTensorTokens(spec, producers->second);
    }
    for (auto& [block, spec] : tailSpecs) plan.tailDrains.push_back({block, std::move(spec)});
    return plan;
}

}  // namespace

WaitInsertionPlan buildGirFrameWaitPlan(Function& function, const GirFrameAnalysis::Result& frames,
                                        const GirFrameHazardAnalysis::Result& hazards) {
    const PotentialMap potentials = buildPotentials(function, frames, hazards);
    if (potentials.empty()) return {};

    DecisionMap decisions;
    for (const auto& [dynamic, _] : potentials) decisions[{dynamic.anchor, dynamic.counter}] = 0;

    TailDecisionMap tailDecisions;
    TailProducerMap tailProducers;
    RequirementMap requirements =
        closeDecisions(function, frames, potentials, decisions, tailDecisions);

    size_t joinSites = 0;
    for (const auto& [site, wait] : decisions) {
        (void)wait;
        if (site.anchor->getParent()->getPredecessors().size() > 1) ++joinSites;
    }
    for (size_t round = 0; round <= joinSites; ++round) {
        PromotionPlan promotions = derivePromotions(requirements, decisions);
        bool changed = false;
        for (const auto& [site, wait] : promotions.anchors) {
            auto found = decisions.find(site);
            if (found == decisions.end() || found->second != wait) {
                decisions[site] = wait;
                changed = true;
            }
        }
        for (const auto& [site, wait] : promotions.tails) {
            auto [found, inserted] = tailDecisions.emplace(site, wait);
            if (inserted || wait < found->second) {
                found->second = std::min(found->second, wait);
                changed = true;
            }
        }
        for (const auto& [site, producers] : promotions.tailProducers)
            tailProducers[site].insert(producers.begin(), producers.end());
        if (!changed) break;
        requirements = closeDecisions(function, frames, potentials, decisions, tailDecisions);
    }

    requirements = simulate(function, frames, potentials, decisions, tailDecisions);
    for (const auto& [site, summary] : requirements) {
        auto decision = decisions.find(site);
        if (decision == decisions.end() || decision->second > summary.strictest)
            report_fatal_error("GIR finite-frame counter flow produced an unsafe wait decision");
    }
    return materializePlan(decisions, tailDecisions, tailProducers, requirements);
}

}  // namespace stinkytofu::waitcnt
