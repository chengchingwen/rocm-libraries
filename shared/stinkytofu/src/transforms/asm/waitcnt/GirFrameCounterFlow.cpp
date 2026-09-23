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
#include <limits>
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

struct IssueKey {
    StinkyInstruction* inst = nullptr;
    GirFrame frame;

    bool operator==(const IssueKey&) const = default;
    bool operator<(const IssueKey& other) const {
        if (inst != other.inst) return std::less<StinkyInstruction*>{}(inst, other.inst);
        return frame < other.frame;
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
    size_t producerIndex = 0;
    int gap = 0;
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


/// The barrier a cross-agent hazard's wait anchors on: the LAST one standing between the two
/// occurrences on the frame graph.  Resolved by position, not by any fence-to-hazard table handed
/// down, so a barrier StinkyTofu placed itself anchors exactly like one it inherited.
/// The barrier this hazard's wait anchors on, via the one shared definition.
std::pair<StinkyInstruction*, GirFrame> enclosingFence(Function& function,
                                                      const GirFrameHazard& hazard,
                                                      const GirFrameAnalysis::Result& frames) {
    (void)function;
    auto found = lastBarrierBefore(frames, hazard.consumerBlock, hazard.consumerFrame,
                                   hazard.consumerIndex);
    if (found.first) return found;
    std::ostringstream message;
    message << "Cross-agent ST frame hazard is discharged by no barrier"
            << " kind=" << static_cast<int>(hazard.kind) << " gap=" << hazard.gap
            << " producerBlock=" << hazard.producerBlock->getLabel()
            << " consumerBlock=" << hazard.consumerBlock->getLabel();
    report_fatal_error(message.str());
}

FeasibleDomainMap computeFeasibleDomains(Function& function,
                                         const GirFrameAnalysis::Result& frames);
bool hazardFeasible(Function& function, const GirFrameAnalysis::Result& frames,
                    const FeasibleDomainMap& feasible, const GirFrameHazard& hazard);

PotentialMap buildPotentials(Function& function, const GirFrameAnalysis::Result& frames,
                             const GirFrameHazardAnalysis::Result& hazards,
                             const std::function<bool(const BasicBlock&)>& covers) {
    PotentialMap result;
    const FeasibleDomainMap feasible = computeFeasibleDomains(function, frames);
    std::set<std::tuple<StinkyInstruction*, GirFrame, CounterKind, StinkyInstruction*, GirFrame,
                        size_t, int>>
        seen;
    std::set<const BasicBlock*> modeled;
    for (BasicBlock& block : function) modeled.insert(&block);
    for (const GirFrameHazard& hazard : hazards.hazards) {
        // GIR models exactly the scheduling region, so an endpoint outside this function is not a
        // hazard of it -- it is a tag riding on a fork that the region never modeled.
        if (!modeled.count(hazard.producerBlock) || !modeled.count(hazard.consumerBlock)) continue;
        if (!covers(*hazard.consumerBlock)) continue;
        if (!hazardFeasible(function, frames, feasible, hazard)) continue;
        CounterKind counter = counterFor(hazard);
        if (counter != CK_DS && counter != CK_Tensor) continue;
        // The frame comes from the match, not from guessing which side the fence sits on: a
        // barrier discharges the hazard wherever on the path it stands, including a block between
        // the two ends.
        StinkyInstruction* anchor = hazard.consumer;
        GirFrame anchorFrame = hazard.consumerFrame;
        if (hazard.crossAgent)
            std::tie(anchor, anchorFrame) = enclosingFence(function, hazard, frames);
        // The span is part of the identity: the hazard analysis deliberately keeps one pair at two
        // distances, and folding them together kept whichever the vector happened to hold first.
        auto key = std::make_tuple(anchor, anchorFrame, counter, hazard.producer,
                                   hazard.producerFrame, hazard.producerIndex, hazard.gap);
        if (!seen.insert(key).second) continue;
        result[{anchor, anchorFrame, counter}].push_back(
            {{hazard.producer, hazard.producerFrame}, hazard.producerIndex, hazard.gap});
    }
    return result;
}

/// Decode from the opcode and the literal, never from SWaitCntData alone: on gfx1250 `dlcnt` is
/// never set, so reading it credited nothing and a second wait was emitted in front of every
/// anchor.  `observedWaitDrains` is the one implementation of this rule.
int observedWait(const StinkyInstruction& inst, CounterKind counter) {
    int counts[CK_Count];
    if (!observedWaitDrains(inst, counts)) return WaitCountSpec::kUnused;
    return counts[counter];
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

int decisionFor(const DecisionMap& decisions, StinkyInstruction* anchor, CounterKind counter) {
    auto found = decisions.find({anchor, counter});
    if (found == decisions.end() || found->second == WaitCountSpec::kUnused) return -1;
    return found->second;
}

size_t indexInBlock(StinkyInstruction& inst) {
    size_t index = 0;
    for (IRBase& node : *inst.getParent()) {
        auto* candidate = dyn_cast<StinkyInstruction>(&node);
        if (!candidate) continue;
        if (candidate == &inst) break;
        ++index;
    }
    return index;
}

/// How many same-counter issues stand at or after `potential`'s producer when control reaches the
/// anchor: 1 is the producer alone, so `waitToDrain` retires it at `n - 1`.
///
/// Walked over the hazard's OWN span rather than read out of a per-node state.  A state keyed by
/// `(instruction, frame)` cannot hold an occurrence a full ring period back -- with ring 2 the read
/// two trips ago and this trip's read are the same key, and the newer one overwrites the older, so
/// the lookup answers 1 for a producer that a full loop of issues has since buried.  The walk
/// crosses those trips and counts them.
///
/// Returns -1 when no path reaches the anchor with the producer still outstanding: an intervening
/// wait already retired it, which is no constraint at all.
///
/// `perPred` collects the MINIMUM over the paths arriving via each predecessor, which is the
/// quantity a per-edge requirement is about.  Reporting instead the predecessor of the single
/// cheapest path makes the answer depend on which of several equal paths the walk happened to
/// reach first -- and that order follows heap addresses, so the same kernel compiled twice got
/// different waits.
template <class RetireAt, class RetireAtEnd>
int walkSpan(const GirFrameAnalysis::Result& frames, CounterKind counter,
             const GirFrameNode& producerNode, size_t producerIndex, int gap,
             const StinkyInstruction* anchor, const GirFrameNode& anchorNode, size_t anchorIndex,
             RetireAt retireAt, RetireAtEnd retireAtEnd,
             std::map<BasicBlock*, int, std::less<BasicBlock*>>* perPred) {
    struct Step {
        GirFrameNode node;
        size_t index = 0;
        int steps = 0;
        int count = 0;
        BasicBlock* pred = nullptr;
    };

    const int span = std::max(0, gap);
    int best = -1;
    std::deque<Step> work{{producerNode, producerIndex + 1, 0, 1, nullptr}};
    // The frame graph is a graph, so the same state is reachable many ways and an unmemoised walk
    // re-expands it exponentially.  Truncating that with a budget is what made the answer depend
    // on exploration order; deduplicating the state makes the walk finite AND complete, because a
    // state already enqueued can contribute nothing a second time.
    std::unordered_set<size_t> visited;
    const auto stateKey = [](const Step& step) {
        size_t key = GirFrameNodeHash{}(step.node);
        for (const size_t part : {step.index, static_cast<size_t>(step.steps),
                                  static_cast<size_t>(step.count),
                                  std::hash<BasicBlock*>{}(step.pred)})
            key = key * 1099511628211ULL ^ part;
        return key;
    };
    visited.insert(stateKey(work.front()));

    while (!work.empty()) {
        Step step = std::move(work.front());
        work.pop_front();

        bool retired = false;
        size_t position = 0;
        for (IRBase& node : *step.node.block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            const size_t here = position++;
            if (here < step.index) continue;

            if (inst == anchor && step.steps == span && here == anchorIndex &&
                step.node == anchorNode) {
                if (best < 0 || step.count < best) best = step.count;
                if (perPred) {
                    auto [entry, inserted] = perPred->emplace(step.pred, step.count);
                    if (!inserted) entry->second = std::min(entry->second, step.count);
                }
                retired = true;  // this path is answered; do not walk past its own anchor
                break;
            }

            if (retireAt(*inst, step.count)) {
                retired = true;
                break;
            }

            if (classifyMemOp(*inst) == counter) ++step.count;
        }
        if (retired) continue;

        if (retireAtEnd(step.node.block, step.count)) continue;
        if (step.steps >= span) continue;

        auto successors = frames.edges.find(step.node);
        if (successors == frames.edges.end()) continue;
        for (const GirFrameNode& successor : successors->second) {
            Step next{successor, 0, step.steps + 1, step.count, step.node.block};
            if (visited.insert(stateKey(next)).second) work.push_back(std::move(next));
        }
    }
    return best;
}

int countAcrossSpan(const GirFrameAnalysis::Result& frames, const DecisionMap& decisions,
                    const TailDecisionMap& tailDecisions, CounterKind counter,
                    const GirFrameNode& producerNode, const Potential& potential,
                    StinkyInstruction* anchor, const GirFrameNode& anchorNode, size_t anchorIndex,
                    std::map<BasicBlock*, int, std::less<BasicBlock*>>* perPred) {
    return walkSpan(
        frames, counter, producerNode, potential.producerIndex, potential.gap, anchor, anchorNode,
        anchorIndex,
        [&](const StinkyInstruction& inst, int count) {
            for (const int keep :
                 {decisionFor(decisions, const_cast<StinkyInstruction*>(&inst), counter),
                  observedWait(inst, counter)})
                if (keep >= 0 && count > keep) return true;
            return false;
        },
        [&](BasicBlock* block, int count) {
            auto tail = tailDecisions.find({block, counter});
            return tail != tailDecisions.end() && count > tail->second;
        },
        perPred);
}

RequirementMap simulate(Function& function, const GirFrameAnalysis::Result& frames,
                        const PotentialMap& potentials, const DecisionMap& decisions,
                        const TailDecisionMap& tailDecisions) {
    (void)function;
    RequirementMap requirements;

    // A `(block, frame)` pair can name several nodes, which differ only by the action that entered
    // them; a producer occurrence names the pair, so every node carrying it is a starting point.
    std::map<std::pair<BasicBlock*, GirFrame>, std::vector<GirFrameNode>> nodesByKey;
    auto remember = [&](const GirFrameNode& node) {
        auto& list = nodesByKey[{node.block, node.frame}];
        for (const GirFrameNode& seen : list)
            if (seen == node) return;
        list.push_back(node);
    };
    for (const auto& [node, successors] : frames.edges) {
        remember(node);
        for (const GirFrameNode& successor : successors) remember(successor);
    }

    for (const auto& [site, list] : potentials) {
        const size_t anchorIndex = indexInBlock(*site.anchor);
        for (const GirFrameNode& anchorNode :
             nodesByKey[{site.anchor->getParent(), site.frame}]) {
            for (const Potential& potential : list) {
                auto producers =
                    nodesByKey.find({potential.producer.inst->getParent(), potential.producer.frame});
                if (producers == nodesByKey.end()) continue;
                for (const GirFrameNode& producerNode : producers->second) {
                    std::map<BasicBlock*, int, std::less<BasicBlock*>> perPred;
                    const int count =
                        countAcrossSpan(frames, decisions, tailDecisions, site.counter, producerNode,
                                        potential, site.anchor, anchorNode, anchorIndex, &perPred);
                    if (count <= 0) continue;
                    for (const auto& [pred, arrived] : perPred)
                        requirements[{site.anchor, site.counter}].record(
                            pred, waitToDrain(site.counter, arrived), potential.producer);
                }
            }
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

/// Closes the decisions by STRENGTHENING only, from "no wait" downwards.
///
/// Both directions reach a fixpoint, but only this one reaches the weakest safe assignment.
/// Seeding every anchor at 0 drains the queue at each anchor before the queue is ever read, so a
/// loop-carried producer is never in the FIFO, its rank is never observed, and 0 re-derives itself
/// -- self-consistent and maximally strong.  Seeding at kUnused leaves the pipeline at its real
/// depth, so each hazard reads the rank its own frame actually holds.
RequirementMap closeDecisions(Function& function, const GirFrameAnalysis::Result& frames,
                              const PotentialMap& potentials, DecisionMap& decisions,
                              const TailDecisionMap& tailDecisions) {
    // Every round that does not close strictly lowers one site, and a site falls at most from its
    // first finite rank to 0, so the depth of the tracked queue bounds the descent.
    size_t depth = 0;
    for (BasicBlock& block : function)
        for (IRBase& node : block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst && (classifyMemOp(*inst) == CK_DS || classifyMemOp(*inst) == CK_Tensor))
                ++depth;
        }
    const size_t rounds = decisions.size() * (depth + 2) + 1;

    for (size_t round = 0; round < rounds; ++round) {
        RequirementMap requirements =
            simulate(function, frames, potentials, decisions, tailDecisions);
        DecisionMap next = decisions;
        for (auto& [site, wait] : next) {
            auto found = requirements.find(site);
            if (found == requirements.end()) continue;
            const int need = found->second.strictest;
            if (need == WaitCountSpec::kUnused) continue;
            if (wait == WaitCountSpec::kUnused || need < wait) wait = need;
        }
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
        if (wait == WaitCountSpec::kUnused) continue;
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

StinkyInstruction* instructionAt(BasicBlock* block, size_t index) {
    size_t position = 0;
    for (IRBase& node : *block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (!inst) continue;
        if (position++ == index) return inst;
    }
    return nullptr;
}

}  // namespace

int girFrameIssuesAcrossSpan(const GirFrameAnalysis::Result& frames, CounterKind counter,
                             const GirFrameNode& producerNode, size_t producerIndex, int span,
                             const GirFrameNode& anchorNode, size_t anchorIndex) {
    StinkyInstruction* anchor = instructionAt(anchorNode.block, anchorIndex);
    if (!anchor) return -1;
    // No planned decisions yet -- a fence is placed before any wait is filled in, so the only
    // thing that can retire the producer early is a wait already standing in the IR.
    return walkSpan(
        frames, counter, producerNode, producerIndex, span, anchor, anchorNode, anchorIndex,
        [&](const StinkyInstruction& inst, int count) {
            const int keep = observedWait(inst, counter);
            return keep >= 0 && count > keep;
        },
        [](BasicBlock*, int) { return false; }, nullptr);
}

int girFrameDistance(const GirFrameAnalysis::Result& frames, const GirFrameNode& from,
                     const GirFrameNode& to) {
    if (from == to) return 0;
    std::unordered_set<GirFrameNode, GirFrameNodeHash> seen{from};
    std::deque<std::pair<GirFrameNode, int>> work{{from, 0}};
    while (!work.empty()) {
        auto [node, steps] = std::move(work.front());
        work.pop_front();
        auto successors = frames.edges.find(node);
        if (successors == frames.edges.end()) continue;
        for (const GirFrameNode& successor : successors->second) {
            if (successor == to) return steps + 1;
            if (!seen.insert(successor).second) continue;
            work.push_back({successor, steps + 1});
        }
    }
    return -1;
}

WaitInsertionPlan buildGirFrameWaitPlan(Function& function, const GirFrameAnalysis::Result& frames,
                                        const GirFrameHazardAnalysis::Result& hazards,
                                        const std::function<bool(const BasicBlock&)>& covers) {
    const PotentialMap potentials = buildPotentials(function, frames, hazards, covers);
    if (potentials.empty()) return {};

    DecisionMap decisions;
    for (const auto& [dynamic, _] : potentials)
        decisions[{dynamic.anchor, dynamic.counter}] = WaitCountSpec::kUnused;

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

    // kUnused is "emit nothing", which is weaker than any count -- never a pass for a site the
    // final state still constrains.
    requirements = simulate(function, frames, potentials, decisions, tailDecisions);
    for (const auto& [site, summary] : requirements) {
        if (summary.strictest == WaitCountSpec::kUnused) continue;
        auto decision = decisions.find(site);
        if (decision == decisions.end() || decision->second == WaitCountSpec::kUnused ||
            decision->second > summary.strictest)
            report_fatal_error("GIR finite-frame counter flow produced an unsafe wait decision");
    }
    return materializePlan(decisions, tailDecisions, tailProducers, requirements);
}

}  // namespace stinkytofu::waitcnt
