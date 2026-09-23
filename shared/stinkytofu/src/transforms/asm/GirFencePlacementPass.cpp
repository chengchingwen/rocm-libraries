/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "stinkytofu/transforms/asm/GirFencePlacementPass.hpp"

#include <algorithm>
#include <climits>
#include <deque>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/transforms/asm/waitcnt/GirFrameCounterFlow.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

#define DEBUG_TYPE "GirFencePlacementPass"

namespace {
using namespace stinkytofu;

constexpr int kWorkgroupBarrierId = -1;

/// Producer occurrences that have executed with no fence crossed since.  A fence publishes
/// EVERYTHING, so the kill is total: tracking discharge per-storage instead is what invents
/// redundant fences, because each one then looks like it covers only its own hazard.
using Live = std::set<size_t>;

/// `(block, frame)` is the finite quotient the frame model already repeats to, which is what
/// bounds this fixpoint exactly as it bounds the wait flow.
struct Node {
    BasicBlock* block = nullptr;
    GirFrame frame;

    bool operator<(const Node& other) const {
        if (block != other.block) return std::less<BasicBlock*>{}(block, other.block);
        return frame < other.frame;
    }
    bool operator==(const Node& other) const = default;
};

bool publishes(const StinkyInstruction& inst) {
    return isBarrier(inst);
}

/// Walk one node, applying gen at producers and the total kill at fences.  `report` sees every
/// consumer whose producer is still live -- an undischarged hazard.
template <class Report>
Live transfer(const Node& node, const std::vector<StinkyInstruction*>& body,
              const GirFrameHazardAnalysis::Result& hazards, const Live& in, Report report) {
    Live live = in;
    for (size_t i = 0; i < body.size(); ++i) {
        StinkyInstruction* inst = body[i];
        if (publishes(*inst)) live.clear();
        for (size_t h = 0; h < hazards.hazards.size(); ++h) {
            const GirFrameHazard& hazard = hazards.hazards[h];
            if (!hazard.crossAgent) continue;
            if (hazard.consumer == inst && hazard.consumerFrame == node.frame && live.count(h))
                report(i, h);
        }
        for (size_t h = 0; h < hazards.hazards.size(); ++h) {
            const GirFrameHazard& hazard = hazards.hazards[h];
            if (!hazard.crossAgent) continue;
            if (hazard.producer == inst && hazard.producerFrame == node.frame) live.insert(h);
        }
    }
    return live;
}

struct FrameCFG {
    std::vector<Node> nodes;
    std::map<Node, size_t> id;
    std::vector<std::vector<size_t>> preds;
    std::vector<std::vector<StinkyInstruction*>> body;
};

/// Every block of the region, not just the schedulable ones: GIR's own CFG roles are spread over
/// `entry`/`skipPGR`/`LoopEndL` after ScaffoldMapPass renames them, and dropping one severs the
/// path a loop-carried producer travels to reach its consumer.
FrameCFG buildFrameCFG(Function& function, const GirFrameAnalysis::Result& frames) {
    FrameCFG cfg;
    for (BasicBlock& block : function) {
        for (const GirFrame& frame : frames.frames(&block)) {
            cfg.id[Node{&block, frame}] = cfg.nodes.size();
            cfg.nodes.push_back(Node{&block, frame});
            std::vector<StinkyInstruction*> body;
            for (IRBase& node : block)
                if (auto* inst = dyn_cast<StinkyInstruction>(&node)) body.push_back(inst);
            cfg.body.push_back(std::move(body));
        }
    }
    // A `(block, frame)` that appears only in the edge set still carries liveness; dropping its
    // edge for want of a node is what leaves a hazard undischarged.
    const auto nodeId = [&cfg](const GirFrameNode& node) {
        auto found = cfg.id.find(Node{node.block, node.frame});
        if (found != cfg.id.end()) return found->second;
        const size_t id = cfg.nodes.size();
        cfg.id[Node{node.block, node.frame}] = id;
        cfg.nodes.push_back(Node{node.block, node.frame});
        std::vector<StinkyInstruction*> body;
        for (IRBase& inner : *node.block)
            if (auto* inst = dyn_cast<StinkyInstruction>(&inner)) body.push_back(inst);
        cfg.body.push_back(std::move(body));
        return id;
    };
    std::vector<std::pair<size_t, size_t>> edges;
    for (const auto& [from, tos] : frames.edges) {
        const size_t source = nodeId(from);
        for (const GirFrameNode& to : tos) edges.emplace_back(source, nodeId(to));
    }
    cfg.preds.resize(cfg.nodes.size());
    for (const auto& [source, sink] : edges) cfg.preds[sink].push_back(source);
    return cfg;
}

/// Least fixpoint of the may-reach set; meet is union, so a producer unfenced on ANY path in is
/// unfenced here.  Finite frames plus a monotone transfer bound the iteration.
std::vector<Live> solve(const FrameCFG& cfg, const GirFrameHazardAnalysis::Result& hazards) {
    std::vector<Live> in(cfg.nodes.size());
    std::vector<Live> out(cfg.nodes.size());
    std::deque<size_t> work;
    for (size_t i = 0; i < cfg.nodes.size(); ++i) work.push_back(i);

    while (!work.empty()) {
        const size_t n = work.front();
        work.pop_front();
        Live merged;
        for (size_t p : cfg.preds[n]) merged.insert(out[p].begin(), out[p].end());
        if (merged == in[n] && !out[n].empty()) continue;
        in[n] = merged;
        Live next = transfer(cfg.nodes[n], cfg.body[n], hazards, in[n], [](size_t, size_t) {});
        if (next == out[n]) continue;
        out[n] = std::move(next);
        for (size_t m = 0; m < cfg.nodes.size(); ++m)
            if (std::find(cfg.preds[m].begin(), cfg.preds[m].end(), n) != cfg.preds[m].end())
                work.push_back(m);
    }
    return in;
}

/// The earliest consumer still seeing its own producer, and every hazard violating there.
bool firstViolation(const FrameCFG& cfg, const GirFrameHazardAnalysis::Result& hazards,
                    const std::vector<Live>& in, size_t& node, size_t& slot,
                    std::vector<size_t>& violating) {
    for (size_t n = 0; n < cfg.nodes.size(); ++n) {
        size_t found = cfg.body[n].size();
        std::vector<size_t> here;
        transfer(cfg.nodes[n], cfg.body[n], hazards, in[n], [&](size_t i, size_t h) {
            if (i < found) {
                found = i;
                here.clear();
            }
            if (i == found) here.push_back(h);
        });
        if (found < cfg.body[n].size()) {
            node = n;
            slot = found;
            violating = std::move(here);
            return true;
        }
    }
    return false;
}

enum class Counter { None, Ds, Tensor };

Counter counterOfProducer(const StinkyInstruction& inst) {
    if (isTensorLoad(inst)) return Counter::Tensor;
    if (isDSRead(inst) || isDSWrite(inst)) return Counter::Ds;
    return Counter::None;
}

bool countsFor(const StinkyInstruction& inst, Counter counter) {
    if (counter == Counter::Ds) return isDSRead(inst) || isDSWrite(inst);
    if (counter == Counter::Tensor) return isTensorLoad(inst);
    return false;
}

/// What is unfenced just before each instruction, so a candidate slot knows what it would
/// discharge -- and therefore which residuals its one wait must be the minimum of.
std::vector<Live> liveBefore(const Node& node, const std::vector<StinkyInstruction*>& body,
                             const GirFrameHazardAnalysis::Result& hazards, const Live& in) {
    std::vector<Live> out(body.size() + 1);
    Live live = in;
    for (size_t i = 0; i < body.size(); ++i) {
        if (publishes(*body[i])) live.clear();
        out[i] = live;
        for (size_t h = 0; h < hazards.hazards.size(); ++h) {
            const GirFrameHazard& hazard = hazards.hazards[h];
            if (hazard.crossAgent && hazard.producer == body[i] &&
                hazard.producerFrame == node.frame)
                live.insert(h);
        }
    }
    out[body.size()] = live;
    return out;
}

waitcnt::CounterKind kindOf(Counter counter) {
    return counter == Counter::Ds ? waitcnt::CK_DS : waitcnt::CK_Tensor;
}

/// Every `GirFrameNode` sharing a `(block, frame)`: an occurrence names the pair, but the frame
/// graph distinguishes nodes by the action that entered them, so all of them carry it.
using NodesByKey = std::map<std::pair<BasicBlock*, GirFrame>, std::vector<GirFrameNode>>;

NodesByKey collectNodes(const GirFrameAnalysis::Result& frames) {
    NodesByKey byKey;
    const auto remember = [&byKey](const GirFrameNode& node) {
        auto& list = byKey[{node.block, node.frame}];
        for (const GirFrameNode& seen : list)
            if (seen == node) return;
        list.push_back(node);
    };
    for (const auto& [node, successors] : frames.edges) {
        remember(node);
        for (const GirFrameNode& successor : successors) remember(successor);
    }
    return byKey;
}

/// What each live hazard already has in flight when this node BEGINS: `base` issues counted over
/// the frame span from its producer, plus the index its in-block tail resumes from.  A residual is
/// then `base` plus the same-counter issues in `[from, slot)`, which is the count the wait pass
/// will measure for a fence at `slot` -- and the part before the block is exactly what a block-local
/// count cannot see.
struct Reach {
    Counter counter = Counter::None;
    int base = 0;
    size_t from = 0;
};

std::optional<Reach> reachAtBlockStart(const GirFrameAnalysis::Result& frames,
                                       const NodesByKey& byKey, const Node& node,
                                       const GirFrameHazard& hazard,
                                       const std::unordered_map<const StinkyInstruction*, size_t>& index) {
    const Counter counter = counterOfProducer(*hazard.producer);
    if (counter == Counter::None) return std::nullopt;

    // A producer in this very node is already positioned; the span is its own distance forward.
    auto producer = index.find(hazard.producer);
    if (hazard.gap == 0 && hazard.producerFrame == node.frame && producer != index.end())
        return Reach{counter, 1, producer->second + 1};

    auto producerNodes = byKey.find({hazard.producerBlock, hazard.producerFrame});
    auto anchorNodes = byKey.find({node.block, node.frame});
    if (producerNodes == byKey.end() || anchorNodes == byKey.end()) return std::nullopt;

    int base = -1;
    for (const GirFrameNode& anchorNode : anchorNodes->second)
        for (const GirFrameNode& producerNode : producerNodes->second) {
            // The hazard's own gap is the frame's answer wherever it lands in this node; a hazard
            // merely passing through is measured to here instead.
            const int span = hazard.consumerBlock == node.block &&
                                     hazard.consumerFrame == node.frame
                                 ? hazard.gap
                                 : waitcnt::girFrameDistance(frames, producerNode, anchorNode);
            if (span < 0) continue;
            const int count = waitcnt::girFrameIssuesAcrossSpan(
                frames, kindOf(counter), producerNode, hazard.producerIndex, span, anchorNode, 0);
            if (count > 0 && (base < 0 || count < base)) base = count;
        }
    if (base < 0) return std::nullopt;
    return Reach{counter, base, 0};
}

/// The slot the cut goes in.  The dataflow fixes how MANY cuts and which slots are legal; inside
/// that freedom the deepest wait wins, because one barrier carries one wait per counter and that
/// wait is the minimum over everything it discharges.
size_t chooseSlot(const GirFrameAnalysis::Result& frames, const NodesByKey& byKey, const Node& node,
                  const std::vector<StinkyInstruction*>& body,
                  const GirFrameHazardAnalysis::Result& hazards,
                  const std::vector<Live>& live, const std::vector<size_t>& violating,
                  size_t consumerSlot) {
    std::unordered_map<const StinkyInstruction*, size_t> index;
    for (size_t i = 0; i < body.size(); ++i) index[body[i]] = i;

    size_t low = 0;
    for (size_t i = consumerSlot; i-- > 0;)
        if (isBranch(*body[i]) || isLabel(*body[i]) || publishes(*body[i])) {
            low = i + 1;
            break;
        }
    // A same-trip producer must already have run, or the cut kills nothing.
    for (size_t h : violating) {
        const GirFrameHazard& hazard = hazards.hazards[h];
        if (hazard.gap != 0) continue;
        auto producer = index.find(hazard.producer);
        if (producer != index.end() && producer->second < consumerSlot)
            low = std::max(low, producer->second + 1);
    }

    std::unordered_map<size_t, std::optional<Reach>> reach;
    const auto reachOf = [&](size_t h) -> const std::optional<Reach>& {
        auto found = reach.find(h);
        if (found == reach.end())
            found = reach.emplace(h, reachAtBlockStart(frames, byKey, node, hazards.hazards[h],
                                                       index))
                        .first;
        return found->second;
    };

    size_t chosen = consumerSlot;
    int best = -1;
    for (size_t slot = low; slot <= consumerSlot; ++slot) {
        int worst = INT_MAX;
        for (size_t h : live[slot]) {
            const std::optional<Reach>& here = reachOf(h);
            if (!here) continue;
            int residual = here->base;
            for (size_t k = here->from; k < slot; ++k)
                if (countsFor(*body[k], here->counter)) ++residual;
            worst = std::min(worst, residual);
        }
        if (worst != INT_MAX && worst > best) {
            best = worst;
            chosen = slot;
        }
    }
    return chosen;
}

class GirFencePlacementPass final : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "GirFencePlacementPass";
    }
    PassID getPassID() const override {
        return &ID;
    }

    PreservedAnalyses run(Function& function, PassContext& passCtx, AnalysisManager& AM) override {
        const auto& frames = AM.getResult<GirFrameAnalysis>(function);
        const auto& hazards = AM.getResult<GirFrameHazardAnalysis>(function);
        if (frames.empty() || hazards.empty()) return PreservedAnalyses::all();

        const GfxArchID archId =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);
        const HwInstDesc* signalDesc = getMCIDByUOp(GFX::s_barrier_signal, archId);
        const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_barrier_wait, archId);
        if (!signalDesc || !waitDesc)
            report_fatal_error("GirFencePlacementPass: no workgroup barrier on this architecture");

        const NodesByKey byKey = collectNodes(frames);
        size_t placed = 0;
        for (;;) {
            FrameCFG cfg = buildFrameCFG(function, frames);
            if (cfg.nodes.empty()) break;
            const std::vector<Live> in = solve(cfg, hazards);
            size_t node = 0;
            size_t slot = 0;
            std::vector<size_t> violating;
            if (!firstViolation(cfg, hazards, in, node, slot, violating)) break;
            slot = chooseSlot(frames, byKey, cfg.nodes[node], cfg.body[node], hazards,
                              liveBefore(cfg.nodes[node], cfg.body[node], hazards, in[node]),
                              violating, slot);
            if (++placed > hazards.hazards.size())
                report_fatal_error("GirFencePlacementPass failed to converge");

            BasicBlock* block = cfg.nodes[node].block;
            AsmIRBuilder builder(*block, archId);
            IRBase* anchor = cfg.body[node][slot];
            StinkyInstruction* signal = builder.create(signalDesc, anchor);
            signal->addSrcReg(StinkyRegister(kWorkgroupBarrierId));
            StinkyInstruction* wait = builder.create(waitDesc, anchor);
            wait->addSrcReg(StinkyRegister(kWorkgroupBarrierId));
            wait->addModifier<CommentData>(CommentData{"GIR fence"});
            // The frame counter flow supplies this barrier's waits, so the token-absence fallback
            // must not also drain it.  `NoWaitCntData` says exactly that and nothing more --
            // marking it a GIR fence would also clear `hasSideEffect` and let it be moved away
            // from the signal it pairs with.
            signal->addModifier<NoWaitCntData>(NoWaitCntData{});
            wait->addModifier<NoWaitCntData>(NoWaitCntData{});
        }
        // The fixpoint minimises the cut; this guarantees the property the wait pass queries.
        // They now ask the same question, so they cannot disagree about whether a barrier stands.
        for (const GirFrameHazard& hazard : hazards.hazards) {
            if (!hazard.crossAgent) continue;
            if (!passCtx.shouldProcessBasicBlock(*hazard.consumerBlock)) continue;
            if (lastBarrierBefore(frames, hazard.consumerBlock, hazard.consumerFrame,
                                  hazard.consumerIndex)
                    .first)
                continue;
            size_t index = 0;
            IRBase* anchor = nullptr;
            for (IRBase& node : *hazard.consumerBlock) {
                auto* inst = dyn_cast<StinkyInstruction>(&node);
                if (!inst) continue;
                if (index++ == hazard.consumerIndex) {
                    anchor = &node;
                    break;
                }
            }
            if (!anchor) continue;
            AsmIRBuilder builder(*hazard.consumerBlock, archId);
            StinkyInstruction* signal = builder.create(signalDesc, anchor);
            signal->addSrcReg(StinkyRegister(kWorkgroupBarrierId));
            StinkyInstruction* wait = builder.create(waitDesc, anchor);
            wait->addSrcReg(StinkyRegister(kWorkgroupBarrierId));
            wait->addModifier<CommentData>(CommentData{"GIR fence (cover)"});
            signal->addModifier<NoWaitCntData>(NoWaitCntData{});
            wait->addModifier<NoWaitCntData>(NoWaitCntData{});
            ++placed;
        }
        // The fixpoint places greedily, so a later cut can subsume an earlier one.  Drop any
        // barrier the remaining set already covers: fewer fences means the copies between them
        // stay in flight together, which is what lets the wait be graded instead of a drain.
        for (bool shrinking = true; shrinking;) {
            shrinking = false;
            // Pair the two halves EXPLICITLY.  Only the wait carries the tag, so collecting
            // "tagged barriers" and pairing them by position takes two unrelated waits for one
            // signal+wait pair and erases both -- which is how the signals outlived their waits.
            std::vector<std::pair<StinkyInstruction*, StinkyInstruction*>> pairs;
            for (BasicBlock& block : function) {
                if (!passCtx.shouldProcessBasicBlock(block)) continue;
                StinkyInstruction* pendingSignal = nullptr;
                for (IRBase& node : block) {
                    auto* inst = dyn_cast<StinkyInstruction>(&node);
                    if (!inst) continue;
                    if (isBarrierSignal(*inst)) {
                        pendingSignal = inst;
                    } else if (isBarrierWait(*inst) && inst->getModifier<CommentData>() &&
                               pendingSignal != nullptr) {
                        pairs.push_back({pendingSignal, inst});
                        pendingSignal = nullptr;
                    }
                }
            }
            for (const auto& [signal, wait] : pairs) {
                BasicBlock* owner = signal->getParent();
                if (wait->getParent() != owner) continue;
                std::vector<IRBase*> saved;
                for (IRBase& node : *owner) saved.push_back(&node);
                owner->removeIR(signal);
                owner->removeIR(wait);
                FrameCFG probe = buildFrameCFG(function, frames);
                size_t n = 0, sl = 0;
                std::vector<size_t> v;
                const bool stillCovered =
                    !probe.nodes.empty() && !firstViolation(probe, hazards, solve(probe, hazards),
                                                            n, sl, v);
                if (stillCovered) {
                    signal->erase();
                    wait->erase();
                    shrinking = true;
                    break;
                }
                // Put the block back exactly as it was: remove+append in the saved order.
                for (IRBase* node : saved) {
                    if (node->getParent()) owner->removeIR(node);
                    owner->appendIR(node);
                }
            }
        }

        if (placed == 0) return PreservedAnalyses::all();
        AM.invalidate(function, preserveCFGAnalyses());
        return preserveCFGAnalyses();
    }
};

char GirFencePlacementPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createGirFencePlacementPass() {
    return std::make_unique<GirFencePlacementPass>();
}
}  // namespace stinkytofu
