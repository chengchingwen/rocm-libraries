/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/AnalysisManager.hpp"
#include "stinkytofu/core/BasicBlock.hpp"

namespace stinkytofu {
struct StinkyInstruction;

inline constexpr const char* kGirFrameContractKey = "gir.frame_contract";

enum class GirActionKind : uint8_t { Other, Read, Copy, Fence, Wmma, WaitCnt };
enum class GirHazardKind : uint8_t { RAW, WAR, WAW };

struct GirGenerationSpec {
    int id = -1;
    int ring = 1;
    int entry = 0;
    int advance = 0;
};

struct GirAccessSpec {
    uint64_t actionId = 0;
    bool isWrite = false;
    int genId = -1;
    int ring = 1;
    int gdelta = 0;
    int absoluteGeneration = -1;
    bool crossAgent = false;
    std::string operand;
    // One set of possible values per independent storage-region axis.
    std::vector<std::vector<int>> regions;
};

struct GirFenceRelationSpec {
    uint64_t fenceAction = 0;
    GirHazardKind kind = GirHazardKind::RAW;
    uint64_t producerAction = 0;
    uint64_t consumerAction = 0;
    int gap = 0;
};

struct GirActionSpec {
    uint64_t id = 0;
    GirActionKind kind = GirActionKind::Other;
    std::vector<size_t> accesses;
};

struct GirFrameContract {
    int version = 0;
    std::map<int, GirGenerationSpec> generations;
    std::map<uint64_t, GirActionSpec> actions;
    std::vector<GirAccessSpec> accesses;
    std::vector<GirFenceRelationSpec> relations;

    static GirFrameContract parse(const std::string& text);
};

struct GirFrame {
    std::vector<std::pair<int, int>> phases;

    int phaseOf(int genId) const;
    void setPhase(int genId, int phase);

    bool operator==(const GirFrame&) const = default;
    bool operator<(const GirFrame& other) const {
        return phases < other.phases;
    }
};

struct GirFrameHash {
    size_t operator()(const GirFrame& frame) const;
};

struct GirFrameNode {
    BasicBlock* block = nullptr;
    GirFrame frame;

    bool operator==(const GirFrameNode&) const = default;
};

struct GirFrameNodeHash {
    size_t operator()(const GirFrameNode& node) const;
};

struct GirAccessOccurrence {
    StinkyInstruction* inst = nullptr;
    BasicBlock* block = nullptr;
    size_t instructionIndex = 0;
    size_t accessIndex = 0;
    GirFrame frame;
    std::vector<int> storage;
};

struct STINKYTOFU_EXPORT GirFrameAnalysis {
    STINKYTOFU_ANALYSIS_KEY("GirFrameAnalysis")

    class Result {
       public:
        GirFrameContract contract;
        std::unordered_map<uint64_t, std::vector<StinkyInstruction*>> actionInstructions;
        std::unordered_map<BasicBlock*, std::vector<GirFrame>> blockFrames;
        std::unordered_map<GirFrameNode, std::vector<GirFrameNode>, GirFrameNodeHash> edges;
        std::vector<GirAccessOccurrence> occurrences;
        std::unordered_set<uint64_t> backEdges;

        bool empty() const {
            return contract.version == 0;
        }
        bool isBackEdge(const BasicBlock* from, const BasicBlock* to) const;
        const std::vector<GirFrame>& frames(const BasicBlock* block) const;
    };

    static Result run(Function& function, AnalysisManager& AM);
};

struct GirFrameHazard {
    GirHazardKind kind = GirHazardKind::RAW;
    StinkyInstruction* producer = nullptr;
    StinkyInstruction* consumer = nullptr;
    BasicBlock* producerBlock = nullptr;
    BasicBlock* consumerBlock = nullptr;
    size_t producerIndex = 0;
    size_t consumerIndex = 0;
    GirFrame producerFrame;
    GirFrame consumerFrame;
    int gap = 0;
    bool crossAgent = false;
    uint64_t producerAction = 0;
    uint64_t consumerAction = 0;
};

struct STINKYTOFU_EXPORT GirFrameHazardAnalysis {
    STINKYTOFU_ANALYSIS_KEY("GirFrameHazardAnalysis")

    struct Result {
        std::vector<GirFrameHazard> hazards;

        bool empty() const {
            return hazards.empty();
        }
    };

    static Result run(Function& function, AnalysisManager& AM);
};

}  // namespace stinkytofu
