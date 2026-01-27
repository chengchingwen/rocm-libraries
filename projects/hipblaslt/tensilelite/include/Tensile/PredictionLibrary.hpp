/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <atomic>
#include <set>
#include <tuple>
#include <vector>

#include <Tensile/UtilsOrigami.hpp>
#include <origami/simulator/tensilelite/formocast_simulator.hpp>

namespace TensileLite
{

    /**
     * \ingroup SolutionLibrary
     *
     * Uses a distance function to select solutions based on benchmarks.
     * Benchmarks are performed to determine the optimal solution at a number of
     * specific sizes. At runtime, we find the benchmarked size that is closest
     * to the size asked for.
     */
    using MinTBInfo         = origami::Formocast::MinTieBreakerInfo;
    using FormoCastPerfInfo = std::tuple<int, double, MinTBInfo>;
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct ProblemPredictionLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        std::unordered_map<int, std::shared_ptr<MySolution>> solutionmap_fc;
        std::unordered_map<int, std::shared_ptr<MySolution>> solutionmap;
        std::vector<origami::config_t>                       origami_config_list;
        std::unordered_map<origami::config_t, int>           origami_config_map;

        mutable std::atomic<bool> lastFindTopRetAll = false;
        bool predictAlgo = 0; // 0: origami, 1: formocast

        static std::string Type()
        {
            return "Prediction";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            if(solutionmap.empty())
                return concatenate(type(), ", solutionmap: empty");
            return concatenate(type(), solutionmap.size());
        }

        static void setupFormoCast(origami::Formocast& formocast, Task& task)
        {
            auto& problem  = task.problem;
            auto& solution = task.solution;

            // GetProblemInfo
            origami::Formocast::ProblemInfo problemInfo;
            problemInfo.M          = solution.calculateDimensionM(problem);
            problemInfo.N          = solution.calculateDimensionN(problem);
            problemInfo.NumBatches = solution.calculateNumBatches(problem);

            problemInfo.K          = problem.boundSize(0);
            problemInfo.transA     = problem.transA();
            problemInfo.transB     = problem.transB();
            problemInfo.bpeA       = problem.a().elementBytes();
            problemInfo.bpeB       = problem.b().elementBytes();
            problemInfo.bpeD       = problem.d().elementBytes();
            problemInfo.bpeCompute = problem.computeTypeElementSize();
            problemInfo.swizzleTensorA = problem.swizzleTensorA();
            problemInfo.swizzleTensorB = problem.swizzleTensorB();
            problemInfo.dataType = problem.f32XdlMathOp() == rocisa::DataType::XFloat32 ?
              origami::data_type_t::XFloat32 :
              datatypeToFormocastDatatype(problem.computeInputType());

            // GetSizeMapping
            auto                                sizeMapping = solution.getSizeMapping();
            origami::Formocast::SizeMapping sm;

            sm.waveNum = sizeMapping.waveNum;

            sm.macroTile[0]      = sizeMapping.macroTile.x;
            sm.macroTile[1]      = sizeMapping.macroTile.y;
            sm.matrixInstruction = sizeMapping.matrixInstruction;

            sm.grvwA = sizeMapping.grvwA;
            sm.grvwB = sizeMapping.grvwB;
            sm.gwvwC = sizeMapping.gwvwC;
            sm.gwvwD = sizeMapping.gwvwD;

            sm.depthU             = sizeMapping.depthU;
            sm.globalSplitU       = solution.calculateAutoGSU(problem, &task.hardware);
            sm.workGroupMapping   = sizeMapping.workGroupMapping;
            sm.globalAccumulation = sizeMapping.globalAccumulation;

            sm.workGroupMappingXCC      = sizeMapping.workGroupMappingXCC;
            sm.workGroupMappingXCCGroup = sizeMapping.workGroupMappingXCCGroup;
            sm.globalSplitUCoalesced    = sizeMapping.globalSplitUCoalesced;
            sm.globalSplitUWorkGroupMappingRoundRobin
                = sizeMapping.globalSplitUWorkGroupMappingRoundRobin;

            sm.CUOccupancy            = sizeMapping.CUOccupancy;
            sm.PrefetchGlobalRead     = sizeMapping.PrefetchGlobalRead;
            sm.MathClocksUnrolledLoop = sizeMapping.MathClocksUnrolledLoop;

            sm.DirectToVgprA      = sizeMapping.DirectToVgprA;
            sm.DirectToVgprB      = sizeMapping.DirectToVgprB;
            sm.NumLoadsCoalescedA = sizeMapping.NumLoadsCoalescedA;
            sm.NumLoadsCoalescedB = sizeMapping.NumLoadsCoalescedB;
            sm.VectorWidthA       = sizeMapping.VectorWidthA;
            sm.VectorWidthB       = sizeMapping.VectorWidthB;
            sm.LocalSplitU        = sizeMapping.LocalSplitU;
            sm.DirectToLdsA       = sizeMapping.DirectToLdsA;
            sm.DirectToLdsB       = sizeMapping.DirectToLdsB;

            sm.waveGroup = sizeMapping.waveGroup;

            formocast.setProblem(problemInfo);
            formocast.setSolution(sm);
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            auto& used_pool = (predictAlgo == 0) ? solutionmap : solutionmap_fc;

            auto indexMatch = used_pool.find(index);
            if(indexMatch != used_pool.end())
                return indexMatch->second;
            return nullptr;
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            auto                        topSolutions = findTopSolutions(problem, hardware, 1);
            std::shared_ptr<MySolution> solution;
            if(!topSolutions.empty())
            {
                solution = topSolutions[0];
            }
            return solution;
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            // TODO- Temp
            if(predictAlgo == 0)
                std::cout << "Entering PredictionLibrary::findAllSolutions(), Algo = Origami" << std::endl;
            else
                std::cout << "Entering PredictionLibrary::findAllSolutions(), Algo = FormoCast" << std::endl;

            bool                    debug = Debug::Instance().printPropertyEvaluation();
            SolutionSet<MySolution> rv;
            if(searchType == SolutionLibrarySearchType::DEFAULT)
                return rv;

            auto& used_pool = (predictAlgo == 0) ? solutionmap : solutionmap_fc;
            for(auto const& row : used_pool)
            {
                if(debug)
                    std::cout << row.second->description() << std::endl;
                rv.insert(row.second);
            }

            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            bool                    debug = Debug::Instance().printPropertyEvaluation();
            SolutionSet<MySolution> rv;
            if(searchType == SolutionLibrarySearchType::DEFAULT)
                return rv;

            auto& used_pool = (predictAlgo == 0) ? solutionmap : solutionmap_fc;
            for(auto const& row : used_pool)
            {
                if(debug)
                    std::cout << row.second->description() << std::endl;
                rv.insert(row.second);
            }

            return rv;
        }

        SolutionVector<MySolution> findTopSolutionsOrigami(MyProblem const& problem,
                                                           Hardware const&  hardware,
                                                           int numSolutions) const
        {
            // TODO- Temp
            std::cout << "Entering PredictionLibrary::findTopSolutionsOrigami()" << std::endl;

            SolutionVector<MySolution> rv;
            size_t                     m     = 1;
            size_t                     n     = 1;
            size_t                     k     = 1;
            size_t                     batch = 1;
            for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
            {
                m *= problem.freeSizeA(i);
            }
            for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
            {
                n *= problem.freeSizeB(i);
            }
            for(size_t i = 0; i < problem.boundIndices().size(); ++i)
            {
                k *= problem.boundSize(i);
            }
            for(size_t i = 0; i < problem.batchIndices().size(); ++i)
            {
                batch *= problem.batchSize(i);
            }

            hip::HipAMDGPU const* pAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);

            const origami::hardware_t& analytical_hardware = *(pAMDGPU->analyticalHardware);
            auto miDataType = datatypeToAnalyticalDatatype(problem.computeInputType());

            if(problem.f32XdlMathOp() == rocisa::DataType::XFloat32) // Check F32 compute type
                miDataType = origami::data_type_t::XFloat32;
            origami::problem_t origami_problem = {
                .size        = {m, n, k},
                .batch       = batch,
                .a_transpose = problem.transA() ? origami::transpose_t::T : origami::transpose_t::N,
                .b_transpose = problem.transB() ? origami::transpose_t::T : origami::transpose_t::N,
                .a_dtype     = datatypeToAnalyticalDatatype(problem.a().dataType()),
                .b_dtype     = datatypeToAnalyticalDatatype(problem.b().dataType()),
                .c_dtype     = datatypeToAnalyticalDatatype(problem.c().dataType()),
                .d_dtype     = datatypeToAnalyticalDatatype(problem.d().dataType()),
                .mi_dtype    = miDataType,
                .a_mx_block_size = 0, // MX Data types come from rocroller
                .b_mx_block_size = 0, // MX Data types come from rocroller
            };

            auto prediction_result = origami::rank_configs(
                origami_problem, *(pAMDGPU->analyticalHardware), origami_config_list);

            for(const auto& r : prediction_result)
            {
                auto mapiter  = origami_config_map.find(r.config);
                auto smapiter = solutionmap.find(mapiter->second);
                if(mapiter != origami_config_map.end() && smapiter != solutionmap.end())
                {
                    auto solution = smapiter->second;
                    if((*solution->hardwarePredicate)(hardware)
                       && (*solution->problemPredicate)(problem))
                    {
                        rv.emplace_back(solution);
                        if(rv.size() == numSolutions)
                        {
                            break;
                        }
                    }
                }
            }

            // can't reach the requested number, means findTop already done its best
            lastFindTopRetAll = (rv.size() < numSolutions);
            return rv;
        }

        SolutionVector<MySolution> findTopSolutionsFormoCast(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             int numSolutions) const
        {
            // TODO- Temp
            //std::cout << "Entering PredictionLibrary::findTopSolutionsFormoCast()" << std::endl;

            bool                           debug = Debug::Instance().printPropertyEvaluation();
            SolutionVector<MySolution>     rv;
            origami::Formocast             formocast;
            std::vector<FormoCastPerfInfo> perfMetric; // sol_idx, micro-s, tieBreakerInfo
            double                         bestMS = std::numeric_limits<double>::max();

            hip::HipAMDGPU const*      pAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
            const origami::hardware_t& analytical_hardware = *(pAMDGPU->analyticalHardware);
            formocast.setHardware(analytical_hardware.arch);

            if(solutionmap_fc.size() == 0) {
                throw std::runtime_error(
                        "[findTopSolutionsFormoCast] No valid solutionmap_fc");
            }

            for(auto const& row : solutionmap_fc)
            {
                int  sol_idx  = row.first;
                auto solution = row.second;

                if(debug)
                {
                    std::cout << solution->description() << ": ";
                }

                if((*solution->hardwarePredicate)(hardware)
                   && (*solution->problemPredicate)(problem))
                {
                    Task task(hardware, problem, *solution);
                    setupFormoCast(formocast, task);
                    // Note:
                    //  formocast.predictedPerformance();
                    //  formocast.getTieBreakerInfo(); // or getMinTieBreakerInfo()
                    auto perf = formocast.predictedPerformance().microSeconds;
                    bestMS    = std::min(bestMS, perf);
                    // auto perf = formocast.predictedPerformance(); //VictorWu
                    // bestMS    = std::min(bestMS, perf.microSeconds);
                    // if(1)//(m_predictionThreshold > 0.0)
                    // {
                    //     std::cout << "=== Performance Info ===" << std::endl;
                    //     std::cout << "perf: " << perf.perf << " us" << std::endl;
                    //     std::cout << "MT0: " << perf.MT0 << std::endl;
                    //     std::cout << "MT1: " << perf.MT1 << std::endl;
                    //     std::cout << "depthU: " << perf.depthU << std::endl;
                    //     std::cout << "NumCUs: " << perf.NumCUs << std::endl;
                    //     std::cout << "WorkGroupMapping: " << perf.WorkGroupMapping << std::endl;
                    //     std::cout << "CUOccupancy: " << perf.CUOccupancy << std::endl;
                    //     std::cout << "GlobalSplitU: " << perf.GlobalSplitU << std::endl;
                    //     std::cout << "LocalSplitU: " << perf.LocalSplitU << std::endl;
                    //     std::cout << "loopCnt: " << perf.loopCnt << std::endl;
                    //     std::cout << "init: " << perf.init << " us" << std::endl;
                    //     std::cout << "preloop: " << perf.preloop << " us" << std::endl;
                    //     std::cout << "Loop: " << perf.loop << " us" << std::endl;
                    //     std::cout << "lsu: " << perf.lsu << " us" << std::endl;
                    //     std::cout << "tail: " << perf.tail << " us" << std::endl;
                    //     std::cout << "math_overall: " << perf.math_overall << std::endl;
                    //     std::cout << "mem_overall: " << perf.mem_overall << std::endl;
                    //     std::cout << "A_loop_hitrate_l1: " << perf.memCosts.cache_hits.L1_hit.tile0HitRate << std::endl;
                    //     std::cout << "B_loop_hitrate_l1: " << perf.memCosts.cache_hits.L1_hit.tile1HitRate << std::endl;
                    //     std::cout << "A_loop_hitrate_l2: " << perf.memCosts.cache_hits.L2_hit.tile0HitRate << std::endl;
                    //     std::cout << "B_loop_hitrate_l2: " << perf.memCosts.cache_hits.L2_hit.tile1HitRate << std::endl;
                    //     std::cout << "A_loop_hitrate_l3: " << perf.memCosts.cache_hits.L3_hit.tile0HitRate << std::endl;
                    //     std::cout << "B_loop_hitrate_l3: " << perf.memCosts.cache_hits.L3_hit.tile1HitRate << std::endl;
                    //     std::cout << "request_l1: "<< perf.memCosts.mem_l1_req << std::endl;
                    //     std::cout << "request_l2: "<< perf.memCosts.mem_l2_req << std::endl;
                    //     std::cout << "request_l3: "<< perf.memCosts.mem_l3_req << std::endl;
                    //     std::cout << "request_hbm: "<< perf.memCosts.mem_hbm_req << std::endl;
                    //     std::cout << "loop_request_l1: "<< perf.memCosts.mem_loop_l1_req << std::endl;
                    //     std::cout << "loop_request_l2: "<< perf.memCosts.mem_loop_l2_req << std::endl;
                    //     std::cout << "loop_request_l3: "<< perf.memCosts.mem_loop_l3_req << std::endl;
                    //     std::cout << "loop_request_hbm: "<< perf.memCosts.mem_loop_hbm_req << std::endl;
                    //     std::cout << "A_MT_request_l1: " << perf.memCosts.MT_A_L1_req << std::endl;
                    //     std::cout << "B_MT_request_l1: " << perf.memCosts.MT_B_L1_req << std::endl;
                    //     std::cout << "A_MT_request_l2: " << perf.memCosts.MT_A_L2_req << std::endl;
                    //     std::cout << "B_MT_request_l2: " << perf.memCosts.MT_B_L2_req << std::endl;
                    //     std::cout << "A_MT_request_l3: " << perf.memCosts.MT_A_L3_req << std::endl;
                    //     std::cout << "B_MT_request_l3: " << perf.memCosts.MT_B_L3_req << std::endl;
                    //     std::cout << "A_MT_request_hbm: " << perf.memCosts.MT_A_hbm_req << std::endl;
                    //     std::cout << "B_MT_request_hbm: " << perf.memCosts.MT_B_hbm_req << std::endl;
                    //     std::cout << "store: " << perf.store << " us" << std::endl;
                    //     std::cout << "gsu: " << perf.gsu << " us" << std::endl;
                    //     std::cout << "num_tiles: " << perf.num_tiles << std::endl;
                    //     std::cout << "=== Performance Info End ===" << std::endl;
                    //     std::cout.unsetf(std::ios::fixed);
                    // }
                    perfMetric.push_back(
                        std::make_tuple(sol_idx, perf, formocast.getMinTieBreakerInfo()));
                        // std::make_tuple(sol_idx, perf.microSeconds, formocast.getMinTieBreakerInfo())); //VictorWu
                }
            }

            if(perfMetric.size() == 0) {
                throw std::runtime_error(
                        "[findTopSolutionsFormoCast] No valid solutions");
            }

            // This sorting function handles m-second first, and then tie-breaker
            auto comp = [&formocast, &bestMS](const FormoCastPerfInfo& metric1,
                                              const FormoCastPerfInfo& metric2) {
                double ms1 = std::get<1>(metric1);
                double ms2 = std::get<1>(metric2);
                // Version 1: first criteria is tie, need to use tie-breaker for the next criteria
                //if(ms1 == ms2)
                // Version 2: we only use tie-breaker to compare for those "faster enough" ones
                if(ms1 < (1.1 * bestMS) && ms2 < (1.1 * bestMS))
                {
                    auto& tbInfo1 = std::get<2>(metric1);
                    auto& tbInfo2 = std::get<2>(metric2);
                    // guard for "strict-weak-ordering" in sorting...
                    if(tbInfo1 == tbInfo2)
                        return ms1 < ms2;

                    // NOTE:
                    // isBetter=true means 2nd is faster, false means Equal
                    // so we need to put tbInfo1 in the 2nd arg.
                    if(formocast.isBetter(tbInfo2, tbInfo1))
                    {
                        return true;
                    }
                    else if(formocast.isBetter(tbInfo1, tbInfo2))
                    {
                        return false;
                    }
                    else
                    {
                        return ms1 < ms2;
                    }
                }
                // sort from: (small -> large) = (faster -> slower) , return TRUE means metric1 is faster
                return ms1 < ms2;
            };
            std::sort(perfMetric.begin(), perfMetric.end(), comp);
            for(int i = 0; i < perfMetric.size(); i++)
            {
                auto solution = solutionmap_fc.at(std::get<0>(perfMetric[i]));
                // std::cout << "perf: " << std::get<1>(perfMetric[i]) << " us" << std::endl; //VictorWu
                rv.emplace_back(solution);
                if(rv.size() == numSolutions)
                {
                    break;
                }
            }

            // can't reach the requested number, means findTop already done its best
            lastFindTopRetAll = (rv.size() < numSolutions);
            return rv;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            // TODO- Temp
            if(predictAlgo == 0)
                return findTopSolutionsOrigami(problem, hardware, numSolutions);
            else
                return findTopSolutionsFormoCast(problem, hardware, numSolutions);
        }

        virtual bool lastFindTopAlreadyRetAll() const override
        {
            return lastFindTopRetAll;
        }

        virtual SolutionVector<MySolution>
            findTopSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        int                           numSolutions) const override
        {
            SolutionVector<MySolution> solutions;
            return solutions;
        }
    };
} // namespace TensileLite
