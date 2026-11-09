/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include <gtest/gtest.h>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "transforms/asm/dag/RegionDAG.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {
constexpr const char* kRingTwoContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0\n"
    "incoming  dst=1 src=1 gen=0 value=1 relative=1\n";

constexpr const char* kFenceContract = "gir-frame-contract\n";

constexpr const char* kRingWrapEntranceContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0\n"
    "incoming  dst=2 src=0 gen=0 value=1\n";

constexpr const char* kForwardingExitContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0\n"
    "incoming  dst=0 src=0 gen=0 value=1 relative=1\n"
    "incoming  dst=1 src=0 gen=0 value=0 relative=1\n";

constexpr const char* kCanonicalSourceAnchorContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0\n"
    "incoming  dst=2 src=0 gen=0 value=1\n";

void setFunctionNumWaves(Function& function, uint32_t numWaves) {
    GemmTileConfig config = function.getGemmTileConfig();
    config.NumWaves = numWaves;
    function.setGemmTileConfig(config);
}

}  // namespace

TEST(GirFrameAnalysisTest, ReconstructsRingOnStCfgAndFindsSameTripRaw) {
    Function function("frame");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(loop, loop);

    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 2, 0, 0, -1, false}}});
    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kRingTwoContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    ASSERT_EQ(frames.frames(loop).size(), 2u);
    EXPECT_TRUE(frames.isBackEdge(loop, loop));
    ASSERT_EQ(frames.occurrences.size(), 4u);

    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    auto raw = std::find_if(
        hazards.hazards.begin(), hazards.hazards.end(), [&](const GirFrameHazard& hazard) {
            return hazard.kind == GirHazardKind::RAW && hazard.producer == copy &&
                   hazard.consumer == read && hazard.gap == 0;
        });
    EXPECT_NE(raw, hazards.hazards.end());
}

TEST(GirFrameAnalysisTest, ForwardingPhiDistinguishesLatchBackEdgeFromDrainExit) {
    Function function("forwarding_exit");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* loop = function.createBasicBlock("loop");
    BasicBlock* drain = function.createBasicBlock("drain");
    function.addEdge(loop, loop);
    function.addEdge(loop, drain);

    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 2, 0, 0, -1, false}}});
    StinkyInstruction* read = createDsReadB128InBlock(drain, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kForwardingExitContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);

    bool sawBackEdgeAdvance = false;
    bool sawUnadvancedExit = false;
    for (const auto& [node, successors] : frames.edges) {
        if (node.block != loop || node.frame.phaseOf(0) != 0) continue;
        for (const GirFrameNode& successor : successors) {
            if (successor.block == loop && successor.frame.phaseOf(0) == 1)
                sawBackEdgeAdvance = true;
            if (successor.block == drain && successor.frame.phaseOf(0) == 0)
                sawUnadvancedExit = true;
        }
    }
    EXPECT_TRUE(sawBackEdgeAdvance);
    EXPECT_TRUE(sawUnadvancedExit);
}

TEST(GirFrameAnalysisTest, LogicalAnchorsSurviveBlockSplittingAndFoldedFirstAction) {
    Function function("split_source_anchor");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* first = function.createBasicBlock("first");
    BasicBlock* second = function.createBasicBlock("second");
    BasicBlock* drain = function.createBasicBlock("drain");
    function.addEdge(first, second);
    function.addEdge(second, drain);

    AsmIRBuilder firstBuilder(*first, GfxArchID::Gfx1250);
    StinkyInstruction* firstAction =
        firstBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));
    firstAction->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Other, {}});
    AsmIRBuilder secondBuilder(*second, GfxArchID::Gfx1250);
    StinkyInstruction* secondAction =
        secondBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));
    secondAction->addModifier<GirActionData>(GirActionData{1, 0, GirActionKind::Other, {}});
    StinkyInstruction* read = createDsReadB128InBlock(drain, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{3, 2, GirActionKind::Ds, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kCanonicalSourceAnchorContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    ASSERT_EQ(frames.frames(drain).size(), 1u);
    EXPECT_EQ(frames.frames(drain).front().phaseOf(0), 1);
}

TEST(GirFrameAnalysisTest, LoopWrapStillUsesUnreducedGenerationSpan) {
    Function function("loop_wrap_generation_span");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(loop, loop);

    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Ds, {GirAccessData{false, 0, 2, 0, 2, -1, false}}});
    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Tensor, {GirAccessData{true, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kRingTwoContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    auto raw = std::find_if(
        hazards.hazards.begin(), hazards.hazards.end(), [&](const GirFrameHazard& hazard) {
            return hazard.kind == GirHazardKind::RAW && hazard.producer == copy &&
                   hazard.consumer == read && hazard.gap == 2;
        });
    EXPECT_NE(raw, hazards.hazards.end());
}

TEST(GirFrameAnalysisTest, EntranceIdentitySurvivesSamePhaseJoin) {
    Function function("same_phase_join");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* left = function.createBasicBlock("left");
    BasicBlock* right = function.createBasicBlock("right");
    BasicBlock* join = function.createBasicBlock("join");
    BasicBlock* destination = function.createBasicBlock("destination");
    function.addEdge(entry, left);
    function.addEdge(entry, right);
    function.addEdge(left, join);
    function.addEdge(right, join);
    function.addEdge(join, destination);

    AsmIRBuilder leftBuilder(*left, GfxArchID::Gfx1250);
    StinkyInstruction* leftAnchor =
        leftBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));
    leftAnchor->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Other, {}});
    AsmIRBuilder rightBuilder(*right, GfxArchID::Gfx1250);
    StinkyInstruction* rightAnchor =
        rightBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));
    rightAnchor->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Other, {}});
    AsmIRBuilder joinBuilder(*join, GfxArchID::Gfx1250);
    joinBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));
    StinkyInstruction* read = createDsReadB128InBlock(destination, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Ds, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kRingWrapEntranceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    ASSERT_EQ(frames.frames(destination).size(), 2u);
    EXPECT_EQ(frames.frames(destination)[0].phaseOf(0), 0);
    EXPECT_EQ(frames.frames(destination)[1].phaseOf(0), 1);
}

namespace {

/// The hazards between two instructions, as kinds.  A WAW survives only when no read orders the
/// write pair, so these four tests read the kind set directly rather than the fences it produces.
std::multiset<GirHazardKind> hazardKindsBetween(AnalysisManager& analyses, Function& function,
                                                const StinkyInstruction* producer,
                                                const StinkyInstruction* consumer) {
    std::multiset<GirHazardKind> kinds;
    const auto& result = analyses.getResult<GirFrameHazardAnalysis>(function);
    for (const GirFrameHazard& hazard : result.hazards)
        if (hazard.producer == producer && hazard.consumer == consumer) kinds.insert(hazard.kind);
    return kinds;
}

}  // namespace

// A read only discharges a WAW on the paths that run it.  The `skip` arm reaches the second write
// without the read, so the write pair keeps its own order there.
TEST(GirFrameAnalysisTest, WawSurvivesTheArmThatSkipsTheRead) {
    Function function("waw_skipped_read");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* consume = function.createBasicBlock("consume");
    BasicBlock* skip = function.createBasicBlock("skip");
    BasicBlock* join = function.createBasicBlock("join");
    function.addEdge(entry, consume);
    function.addEdge(entry, skip);
    function.addEdge(consume, join);
    function.addEdge(skip, join);

    StinkyInstruction* first = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    first->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 2, -1, 0, 0, true}}});
    StinkyInstruction* read = createDsReadB128InBlock(consume, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{false, 0, 2, -1, 0, 0, true}}});
    StinkyInstruction* second = createTensorLoadInBlock(join, GfxArchID::Gfx1250, 16, 24);
    second->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Tensor, {GirAccessData{true, 0, 2, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    EXPECT_EQ(hazardKindsBetween(analyses, function, first, second),
              (std::multiset{GirHazardKind::WAW}));
    EXPECT_EQ(hazardKindsBetween(analyses, function, first, read),
              (std::multiset{GirHazardKind::RAW}));
    EXPECT_EQ(hazardKindsBetween(analyses, function, read, second),
              (std::multiset{GirHazardKind::WAR}));
}

// With the read on every path, RAW then WAR already order the two writes.  Every read still keeps
// its own RAW: the first one does not stand in for the second.
TEST(GirFrameAnalysisTest, InterveningReadDropsTheWawAndKeepsEveryRaw) {
    Function function("waw_ordered_by_read");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* first = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    first->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 2, -1, 0, 0, true}}});
    StinkyInstruction* early = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    early->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{false, 0, 2, -1, 0, 0, true}}});
    StinkyInstruction* late = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 4, 24);
    late->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Ds, {GirAccessData{false, 0, 2, -1, 0, 0, true}}});
    StinkyInstruction* second = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 16, 24);
    second->addModifier<GirActionData>(
        GirActionData{3, 3, GirActionKind::Tensor, {GirAccessData{true, 0, 2, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    EXPECT_TRUE(hazardKindsBetween(analyses, function, first, second).empty());
    EXPECT_EQ(hazardKindsBetween(analyses, function, first, early),
              (std::multiset{GirHazardKind::RAW}));
    EXPECT_EQ(hazardKindsBetween(analyses, function, first, late),
              (std::multiset{GirHazardKind::RAW}));
    EXPECT_EQ(hazardKindsBetween(analyses, function, early, second),
              (std::multiset{GirHazardKind::WAR}));
    EXPECT_EQ(hazardKindsBetween(analyses, function, late, second),
              (std::multiset{GirHazardKind::WAR}));
}

// The read names generation 1, the writes name generation 0.  Sharing an operand is not sharing a
// slot, so this read orders nothing.
TEST(GirFrameAnalysisTest, AReadOfAnotherGenerationDoesNotOrderTheWaw) {
    Function function("waw_other_generation");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* first = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    first->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 2, -1, 0, 0, true}}});
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{false, 0, 2, -1, 0, 1, true}}});
    StinkyInstruction* second = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 16, 24);
    second->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Tensor, {GirAccessData{true, 0, 2, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    EXPECT_EQ(hazardKindsBetween(analyses, function, first, second),
              (std::multiset{GirHazardKind::WAW}));
    EXPECT_TRUE(hazardKindsBetween(analyses, function, first, read).empty());
}

// The claim is about rotating storage.  A ring of 1 has one slot and keeps its WAW.
TEST(GirFrameAnalysisTest, RingOneKeepsItsWawThroughTheRead) {
    Function function("waw_ring_one");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* first = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    first->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* second = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 16, 24);
    second->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Tensor, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    EXPECT_EQ(hazardKindsBetween(analyses, function, first, second),
              (std::multiset{GirHazardKind::WAW}));
}
