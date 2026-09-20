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
    "GIR_FRAME_CONTRACT_V1\n"
    "GEN 0 2 0 1\n"
    "ACTION 0 copy\n"
    "ACCESS 0 1 0 2 0 -1 0 A -\n"
    "ACTION 1 read\n"
    "ACCESS 1 0 0 2 0 -1 0 A -\n";

constexpr const char* kDiamondContract =
    "GIR_FRAME_CONTRACT_V1\n"
    "ACTION 0 copy\n"
    "ACCESS 0 1 -1 1 0 0 0 A -\n"
    "ACTION 1 read\n"
    "ACCESS 1 0 -1 1 0 0 0 A -\n";

constexpr const char* kWarContract =
    "GIR_FRAME_CONTRACT_V1\n"
    "ACTION 0 read\n"
    "ACCESS 0 0 -1 1 0 0 0 A -\n"
    "ACTION 1 copy\n"
    "ACCESS 1 1 -1 1 0 0 0 A -\n";

constexpr const char* kFenceContract =
    "GIR_FRAME_CONTRACT_V1\n"
    "ACTION 0 copy\n"
    "ACCESS 0 1 -1 1 0 0 1 A -\n"
    "ACTION 1 fence\n"
    "ACTION 2 read\n"
    "ACCESS 2 0 -1 1 0 0 1 A -\n"
    "REL 1 RAW 0 2 0\n";

constexpr const char* kPromotionContract =
    "GIR_FRAME_CONTRACT_V1\n"
    "ACTION 0 copy\n"
    "ACCESS 0 1 -1 1 0 0 0 MX -\n"
    "ACTION 1 copy\n"
    "ACCESS 1 1 -1 1 0 0 0 A -\n"
    "ACTION 2 read\n"
    "ACCESS 2 0 -1 1 0 0 0 MX -\n"
    "ACTION 3 read\n"
    "ACCESS 3 0 -1 1 0 0 0 A -\n";

}  // namespace

TEST(GirFrameAnalysisTest, ReconstructsRingOnStCfgAndFindsSameTripRaw) {
    Function function("frame");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(loop, loop);

    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(GirActionData{0});
    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{1});
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
    dag::addGirFrameHazardEdges(graph, hazards);

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
    copy->addModifier<GirActionData>(GirActionData{0});
    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{1});
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
    copy->addModifier<GirActionData>(GirActionData{0});
    StinkyInstruction* read = createDsReadB128InBlock(join, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{1});
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
    read->addModifier<GirActionData>(GirActionData{0});
    StinkyInstruction* copy = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(GirActionData{1});
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
    copy->addModifier<GirActionData>(GirActionData{0});
    AsmIRBuilder builder(*block, GfxArchID::Gfx1250);
    StinkyInstruction* fence = builder.createFence();
    fence->addModifier<GirActionData>(GirActionData{1});
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{2});
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
    producer->addModifier<GirActionData>(GirActionData{0});
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
    consumer->addModifier<GirActionData>(GirActionData{1});
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
    mx->addModifier<GirActionData>(GirActionData{0});
    StinkyInstruction* a = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 16, 24);
    a->addModifier<GirActionData>(GirActionData{1});
    createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 32, 40);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 48, 56);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 64, 72);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 80, 88);

    StinkyInstruction* mxRead = createDsReadB128InBlock(join, GfxArchID::Gfx1250, 0, 20);
    mxRead->addModifier<GirActionData>(GirActionData{2});
    StinkyInstruction* aRead = createDsReadB128InBlock(join, GfxArchID::Gfx1250, 4, 24);
    aRead->addModifier<GirActionData>(GirActionData{3});
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
