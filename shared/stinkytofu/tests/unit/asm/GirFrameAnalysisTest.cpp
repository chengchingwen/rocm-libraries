/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include <gtest/gtest.h>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/transforms/asm/GirWaitCntInsertionPass.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "transforms/asm/dag/RegionDAG.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {
constexpr const char* kRingTwoContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0 advance=1\n";

constexpr const char* kDiamondContract =
    "gir-frame-contract\n";

constexpr const char* kWarContract =
    "gir-frame-contract\n";

constexpr const char* kFenceContract =
    "gir-frame-contract\n"
    "rel 1  RAW  producer=0 consumer=2\n";

constexpr const char* kPromotionContract =
    "gir-frame-contract\n";

constexpr const char* kEntrancePhiContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0 advance=1\n"
    "incoming  dst=2 src=0 gen=0 value=1\n";

constexpr const char* kRingWrapEntranceContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0 advance=1\n"
    "incoming  dst=2 src=0 gen=0 value=1\n";

constexpr const char* kForwardingExitContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0 advance=1\n"
    "transfer  dst=0 src=0 gen=0 delta=1\n"
    "transfer  dst=1 src=0 gen=0 delta=0\n";

constexpr const char* kCanonicalSourceAnchorContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0 advance=1\n"
    "incoming  dst=2 src=0 gen=0 value=1\n";

constexpr const char* kAgentRelativeRegionsContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0 advance=1\n"
    "incoming  dst=2 src=0 gen=0 value=1\n"
    "rel 2  RAW  producer=0 consumer=3  gap=1\n";

}  // namespace

TEST(GirFrameAnalysisTest, ReconstructsRingOnStCfgAndFindsSameTripRaw) {
    Function function("frame");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(loop, loop);

    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, "A", 2, 0, 0, -1, false, -1}}});
    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Read, {GirAccessData{false, "A", 2, 0, 0, -1, false, -1}}});
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
    copy->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, "A", 2, 0, 0, -1, false, -1}}});
    StinkyInstruction* read = createDsReadB128InBlock(drain, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Read, {GirAccessData{false, "A", 2, 0, 0, -1, false, -1}}});
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
    read->addModifier<GirActionData>(GirActionData{3, 2, GirActionKind::Read, {GirAccessData{false, "A", 2, 0, 0, -1, false, -1}}});
    function.setStringMetaData(kGirFrameContractKey, kCanonicalSourceAnchorContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    ASSERT_EQ(frames.frames(drain).size(), 1u);
    EXPECT_EQ(frames.frames(drain).front().phaseOf(0), 1);
}

TEST(GirFrameAnalysisTest, AgentRelativeReadWaitsForEveryPhysicalRegionProducer) {
    Function function("agent_relative_regions");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* drain = function.createBasicBlock("drain");
    function.addEdge(entry, drain);

    StinkyInstruction* first = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    first->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, "A", 2, -1, 0, 0, true, 0}}});
    StinkyInstruction* second = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 16, 24);
    second->addModifier<GirActionData>(GirActionData{1, 0, GirActionKind::Copy, {GirAccessData{true, "A", 2, -1, 0, 0, true, 1}}});
    AsmIRBuilder builder(*drain, GfxArchID::Gfx1250);
    StinkyInstruction* fence = builder.createFence();
    fence->addModifier<GirActionData>(GirActionData{2, 2, GirActionKind::Fence, {}});
    StinkyInstruction* read = createDsReadB128InBlock(drain, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{3, 2, GirActionKind::Read, {GirAccessData{false, "A", 2, 0, -1, -1, true, 0}, GirAccessData{false, "A", 2, 0, -1, -1, true, 1}}});
    function.setStringMetaData(kGirFrameContractKey, kAgentRelativeRegionsContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    StinkyInstruction* beforeFence = nullptr;
    for (IRBase& node : *drain) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == fence) break;
        beforeFence = inst;
    }
    ASSERT_NE(beforeFence, nullptr);
    const auto* wait = beforeFence->getModifier<SWaitTensorCntData>();
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->tlcnt, 0);
}

TEST(GirFrameAnalysisTest, HazardBecomesAnOrdinaryDagEdge) {
    Function function("dag");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* block = function.createBasicBlock("entry");
    StinkyInstruction* copy = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);

    dag::RegionDAG graph = dag::buildRegisterDependencyDAG({copy, read});
    GirFrameHazardAnalysis::Result hazards;
    hazards.hazards.push_back({GirHazardKind::RAW, copy, read, block, block, 0, 1, GirFrame{},
                               GirFrame{}, 0, false, 0, 1});
    GirFrameAnalysis::Result frames;
    dag::addGirFrameHazardEdges(graph, hazards, frames);

    ASSERT_TRUE(graph.graph[0].contains(1));
    EXPECT_EQ(graph.nodes[0].inDegree, 0u);
    EXPECT_EQ(graph.nodes[1].inDegree, 1u);
}

TEST(GirFrameAnalysisTest, FrameSchedulingDoesNotMaterializeLegacyMemoryTokenRegisters) {
    Function function("frame_without_tokens");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* block = function.createBasicBlock("entry");
    StinkyInstruction* copy = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    copy->addModifier<MemTokenData>(MemTokenData{std::vector<int>{7}});
    read->addModifier<MemTokenData>(MemTokenData{std::vector<int>{7}});

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createStinkyBuildImplicitDependencyPass(false)->run(function, context, analyses);

    auto hasLdsRegister = [](const auto& regs) {
        return std::any_of(regs.begin(), regs.end(), [](const StinkyRegister& reg) {
            return reg.isRegister() && reg.reg.type == RegType::LDS;
        });
    };
    EXPECT_FALSE(hasLdsRegister(copy->getSrcRegs()));
    EXPECT_FALSE(hasLdsRegister(copy->getDestRegs()));
    EXPECT_FALSE(hasLdsRegister(read->getSrcRegs()));
    EXPECT_FALSE(hasLdsRegister(read->getDestRegs()));
}

TEST(GirFrameAnalysisTest, FrameSchedulingDoesNotAddLegacyTokenWalls) {
    Function function("frame_without_token_walls");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* block = function.createBasicBlock("entry");
    AsmIRBuilder builder(*block, GfxArchID::Gfx1250);
    StinkyInstruction* first = builder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));
    StinkyInstruction* fence = builder.createFence();
    StinkyInstruction* last = builder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));
    first->addModifier<MemTokenData>(MemTokenData{std::vector<int>{7}});
    fence->addModifier<OrderTokenData>(OrderTokenData{std::vector<int>{7}});
    last->addModifier<MemTokenData>(MemTokenData{std::vector<int>{7}});

    dag::RegionDAG legacy = dag::buildRegisterDependencyDAG({first, fence, last}, true);
    EXPECT_TRUE(legacy.graph[0].contains(1));
    EXPECT_TRUE(legacy.graph[1].contains(2));

    dag::RegionDAG frame = dag::buildRegisterDependencyDAG({first, fence, last}, false);
    EXPECT_TRUE(frame.graph[0].empty());
    EXPECT_TRUE(frame.graph[1].empty());
    EXPECT_TRUE(frame.graph[2].empty());
}

TEST(GirFrameAnalysisTest, FrameWaitPassInsertsTensorWaitAtTaggedConsumer) {
    Function function("wait");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(loop, loop);
    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, "A", 2, 0, 0, -1, false, -1}}});
    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Read, {GirAccessData{false, "A", 2, 0, 0, -1, false, -1}}});
    function.setStringMetaData(kGirFrameContractKey, kRingTwoContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    StinkyInstruction* beforeRead = nullptr;
    for (IRBase& node : *loop) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == read) break;
        beforeRead = inst;
    }
    ASSERT_NE(beforeRead, nullptr);
    const auto* wait = beforeRead->getModifier<SWaitTensorCntData>();
    ASSERT_NE(wait, nullptr);
    EXPECT_GE(wait->tlcnt, 0);
}

TEST(GirFrameAnalysisTest, JoinIgnoresAbsentPathWithoutInventingPromotion) {
    Function function("diamond");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* producing = function.createBasicBlock("producing");
    BasicBlock* empty = function.createBasicBlock("empty");
    BasicBlock* join = function.createBasicBlock("join");
    function.addEdge(entry, producing);
    function.addEdge(entry, empty);
    function.addEdge(producing, join);
    function.addEdge(empty, join);

    StinkyInstruction* copy = createTensorLoadInBlock(producing, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, "A", 1, -1, 0, 0, false, -1}}});
    StinkyInstruction* read = createDsReadB128InBlock(join, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Read, {GirAccessData{false, "A", 1, -1, 0, 0, false, -1}}});
    function.setStringMetaData(kGirFrameContractKey, kDiamondContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    bool producingWait = false;
    for (IRBase& node : *producing) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        producingWait |= inst && inst->getModifier<SWaitTensorCntData>();
    }
    bool joinWait = false;
    for (IRBase& node : *join) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        joinWait |= inst && inst->getModifier<SWaitTensorCntData>();
    }
    EXPECT_FALSE(producingWait);
    EXPECT_TRUE(joinWait);
}

TEST(GirFrameAnalysisTest, FrameWarRetiresVacatingDsRead) {
    Function function("war");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* block = function.createBasicBlock("entry");
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Read, {GirAccessData{false, "A", 1, -1, 0, 0, false, -1}}});
    StinkyInstruction* copy = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Copy, {GirAccessData{true, "A", 1, -1, 0, 0, false, -1}}});
    function.setStringMetaData(kGirFrameContractKey, kWarContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    StinkyInstruction* beforeCopy = nullptr;
    for (IRBase& node : *block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == copy) break;
        beforeCopy = inst;
    }
    ASSERT_NE(beforeCopy, nullptr);
    const auto* wait = beforeCopy->getModifier<SWaitCntData>();
    ASSERT_NE(wait, nullptr);
    EXPECT_GE(wait->dlcnt, 0);
}

TEST(GirFrameAnalysisTest, CrossAgentWaitAnchorsAtVirtualFence) {
    Function function("fence");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* block = function.createBasicBlock("entry");
    StinkyInstruction* copy = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, "A", 1, -1, 0, 0, true, -1}}});
    AsmIRBuilder builder(*block, GfxArchID::Gfx1250);
    StinkyInstruction* fence = builder.createFence();
    fence->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Fence, {}});
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{2, 2, GirActionKind::Read, {GirAccessData{false, "A", 1, -1, 0, 0, true, -1}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    StinkyInstruction* beforeFence = nullptr;
    StinkyInstruction* beforeRead = nullptr;
    for (IRBase& node : *block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == fence) beforeFence = beforeRead;
        if (inst == read) break;
        beforeRead = inst;
    }
    ASSERT_NE(beforeFence, nullptr);
    EXPECT_NE(beforeFence->getModifier<SWaitTensorCntData>(), nullptr);
    EXPECT_EQ(beforeRead, fence);
}

TEST(GirFrameAnalysisTest, ClonedFenceActionAnchorsWaitOnEachCfgPath) {
    Function function("cloned_fence");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* left = function.createBasicBlock("left");
    BasicBlock* right = function.createBasicBlock("right");
    function.addEdge(entry, left);
    function.addEdge(entry, right);

    StinkyInstruction* copy = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, "A", 1, -1, 0, 0, true, -1}}});

    AsmIRBuilder leftBuilder(*left, GfxArchID::Gfx1250);
    StinkyInstruction* leftFence = leftBuilder.createFence();
    leftFence->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Fence, {}});
    StinkyInstruction* leftRead = createDsReadB128InBlock(left, GfxArchID::Gfx1250, 0, 20);
    leftRead->addModifier<GirActionData>(GirActionData{2, 2, GirActionKind::Read, {GirAccessData{false, "A", 1, -1, 0, 0, true, -1}}});

    AsmIRBuilder rightBuilder(*right, GfxArchID::Gfx1250);
    StinkyInstruction* rightFence = rightBuilder.createFence();
    rightFence->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Fence, {}});
    StinkyInstruction* rightRead = createDsReadB128InBlock(right, GfxArchID::Gfx1250, 0, 20);
    rightRead->addModifier<GirActionData>(GirActionData{2, 2, GirActionKind::Read, {GirAccessData{false, "A", 1, -1, 0, 0, true, -1}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    auto waitImmediatelyBefore = [](BasicBlock* block, StinkyInstruction* anchor) {
        StinkyInstruction* prior = nullptr;
        for (IRBase& node : *block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst == anchor) break;
            prior = inst;
        }
        return prior ? prior->getModifier<SWaitTensorCntData>() : nullptr;
    };
    EXPECT_NE(waitImmediatelyBefore(left, leftFence), nullptr);
    EXPECT_NE(waitImmediatelyBefore(right, rightFence), nullptr);
}

TEST(GirFrameAnalysisTest, EntrancePhiSurvivesUntaggedScaffoldBlocks) {
    Function function("entrance_phi");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* bypass = function.createBasicBlock("bypass");
    BasicBlock* loop = function.createBasicBlock("loop");
    BasicBlock* drain = function.createBasicBlock("drain");
    function.addEdge(entry, bypass);
    function.addEdge(entry, loop);
    function.addEdge(bypass, drain);
    function.addEdge(loop, loop);
    function.addEdge(loop, drain);

    StinkyInstruction* prologueCopy = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    prologueCopy->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, "A", 2, -1, 0, 0, false, -1}}});
    AsmIRBuilder bypassBuilder(*bypass, GfxArchID::Gfx1250);
    bypassBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));

    StinkyInstruction* loopCopy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 16, 24);
    loopCopy->addModifier<GirActionData>(GirActionData{3, 3, GirActionKind::Copy, {GirAccessData{true, "A", 2, 0, 1, -1, false, -1}}});
    StinkyInstruction* drainRead = createDsReadB128InBlock(drain, GfxArchID::Gfx1250, 0, 20);
    drainRead->addModifier<GirActionData>(GirActionData{2, 2, GirActionKind::Read, {GirAccessData{false, "A", 2, 0, -1, -1, false, -1}}});
    function.setStringMetaData(kGirFrameContractKey, kEntrancePhiContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    ASSERT_EQ(frames.frames(drain).size(), 2u);
    EXPECT_TRUE(std::any_of(frames.frames(drain).begin(), frames.frames(drain).end(),
                            [](const GirFrame& frame) { return frame.phaseOf(0) == 1; }));
    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    EXPECT_TRUE(std::any_of(
        hazards.hazards.begin(), hazards.hazards.end(), [&](const GirFrameHazard& hazard) {
            return hazard.kind == GirHazardKind::RAW && hazard.producer == prologueCopy &&
                   hazard.consumer == drainRead;
        }));

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    createGirWaitCntInsertionPass()->run(function, context, analyses);
    StinkyInstruction* beforeRead = nullptr;
    for (IRBase& node : *drain) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == drainRead) break;
        beforeRead = inst;
    }
    ASSERT_NE(beforeRead, nullptr);
    EXPECT_NE(beforeRead->getModifier<SWaitTensorCntData>(), nullptr);
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
    read->addModifier<GirActionData>(GirActionData{2, 2, GirActionKind::Read, {GirAccessData{false, "A", 2, 0, 0, -1, false, -1}}});
    function.setStringMetaData(kGirFrameContractKey, kRingWrapEntranceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    ASSERT_EQ(frames.frames(destination).size(), 2u);
    EXPECT_EQ(frames.frames(destination)[0].phaseOf(0), 0);
    EXPECT_EQ(frames.frames(destination)[1].phaseOf(0), 1);
}

TEST(GirFrameAnalysisTest, TripDomainExcludesShortPrefetchPathFromSteadyWaitRank) {
    Function function("trip_domain");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* shortPath = function.createBasicBlock("label_skipPGR2_1");
    BasicBlock* fullPath = function.createBasicBlock("entry_1");
    BasicBlock* guard = function.createBasicBlock("label_openLoopL_1");
    BasicBlock* loop = function.createBasicBlock("label_LoopBeginL");
    BasicBlock* exit = function.createBasicBlock("label_LoopEndL");
    function.addEdge(entry, shortPath);
    function.addEdge(entry, fullPath);
    function.addEdge(shortPath, guard);
    function.addEdge(fullPath, guard);
    function.addEdge(guard, exit);
    function.addEdge(guard, loop);

    StinkyInstruction* producer = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    producer->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, "A", 1, -1, 0, 0, false, -1}}});
    createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 16, 24);
    createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 32, 40);
    {
        AsmIRBuilder builder(*entry, GfxArchID::Gfx1250);
        StinkyInstruction* compare =
            builder.create(getMCIDByUOp(GFX::s_cmp_eq_u32, GfxArchID::Gfx1250));
        compare->addSrcReg(StinkyRegister("s", 14, 1));
        compare->addSrcReg(StinkyRegister("0x1"));
        StinkyInstruction* branch =
            builder.create(getMCIDByUOp(GFX::s_cbranch_scc1, GfxArchID::Gfx1250));
        branch->addModifier<LabelData>(LabelData{"label_skipPGR2_1"});
    }

    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 48, 56);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 64, 72);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 80, 88);
    {
        AsmIRBuilder builder(*guard, GfxArchID::Gfx1250);
        StinkyInstruction* compare =
            builder.create(getMCIDByUOp(GFX::s_cmp_le_u32, GfxArchID::Gfx1250));
        compare->addSrcReg(StinkyRegister("s", 14, 1));
        compare->addSrcReg(StinkyRegister("0x2"));
        StinkyInstruction* branch =
            builder.create(getMCIDByUOp(GFX::s_cbranch_scc1, GfxArchID::Gfx1250));
        branch->addModifier<LabelData>(LabelData{"label_LoopEndL"});
    }

    StinkyInstruction* consumer = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    consumer->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Read, {GirAccessData{false, "A", 1, -1, 0, 0, false, -1}}});
    function.setStringMetaData(kGirFrameContractKey, kDiamondContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    StinkyInstruction* beforeConsumer = nullptr;
    for (IRBase& node : *loop) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == consumer) break;
        beforeConsumer = inst;
    }
    ASSERT_NE(beforeConsumer, nullptr);
    const auto* wait = beforeConsumer->getModifier<SWaitTensorCntData>();
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->tlcnt, 5);
}

TEST(GirFrameAnalysisTest, PredecessorPromotionRerunsDownstreamCounterflow) {
    Function function("promotion");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* shortPath = function.createBasicBlock("short");
    BasicBlock* fullPath = function.createBasicBlock("full");
    BasicBlock* join = function.createBasicBlock("join");
    function.addEdge(entry, shortPath);
    function.addEdge(entry, fullPath);
    function.addEdge(shortPath, join);
    function.addEdge(fullPath, join);

    StinkyInstruction* mx = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    mx->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, "MX", 1, -1, 0, 0, false, -1}}});
    StinkyInstruction* a = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 16, 24);
    a->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Copy, {GirAccessData{true, "A", 1, -1, 0, 0, false, -1}}});
    createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 32, 40);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 48, 56);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 64, 72);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 80, 88);

    StinkyInstruction* mxRead = createDsReadB128InBlock(join, GfxArchID::Gfx1250, 0, 20);
    mxRead->addModifier<GirActionData>(GirActionData{2, 2, GirActionKind::Read, {GirAccessData{false, "MX", 1, -1, 0, 0, false, -1}}});
    StinkyInstruction* aRead = createDsReadB128InBlock(join, GfxArchID::Gfx1250, 4, 24);
    aRead->addModifier<GirActionData>(GirActionData{3, 3, GirActionKind::Read, {GirAccessData{false, "A", 1, -1, 0, 0, false, -1}}});
    function.setStringMetaData(kGirFrameContractKey, kPromotionContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    const SWaitTensorCntData* shortWait = nullptr;
    for (IRBase& node : *shortPath) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst && inst->getModifier<SWaitTensorCntData>())
            shortWait = inst->getModifier<SWaitTensorCntData>();
    }
    ASSERT_NE(shortWait, nullptr);
    EXPECT_EQ(shortWait->tlcnt, 1);

    std::vector<int> joinWaits;
    for (IRBase& node : *join) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst && inst->getModifier<SWaitTensorCntData>())
            joinWaits.push_back(inst->getModifier<SWaitTensorCntData>()->tlcnt);
    }
    EXPECT_EQ(joinWaits, (std::vector<int>{5, 4}));
}
