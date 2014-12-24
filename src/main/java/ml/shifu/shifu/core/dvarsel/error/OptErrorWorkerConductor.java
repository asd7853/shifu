package ml.shifu.shifu.core.dvarsel.error;
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.dvarsel.AbstractWorkerConductor;
import ml.shifu.shifu.core.dvarsel.VarSelMasterResult;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;
import ml.shifu.shifu.core.dvarsel.wrapper.ValidationConductor;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.collections.CollectionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created on 12/4/2014.
 */
public class OptErrorWorkerConductor extends AbstractWorkerConductor {

    private static final Logger LOG = LoggerFactory.getLogger(OptErrorWorkerConductor.class);

    private List<ColumnConfig> candidates;
    private Set<Integer> baseColumnSet;

    public OptErrorWorkerConductor(ModelConfig modelConfig, List<ColumnConfig> columnConfigList) {
        super(modelConfig, columnConfigList);

        this.candidates = new ArrayList<ColumnConfig>();
        for ( ColumnConfig columnConfig : columnConfigList ) {
            if ( CommonUtils.isGoodCandidate(columnConfig) && !columnConfig.isForceSelect() ) {
                candidates.add(columnConfig);
            }
        }
        LOG.info("Candidate size: {}", candidates.size());
    }

    @Override
    public void consumeMasterResult(VarSelMasterResult masterResult) {
        baseColumnSet = new HashSet<Integer>(masterResult.getColumnIdList());
    }

    @Override
    public VarSelWorkerResult generateVarSelResult() {
        double maxErrorGain = - Double.MAX_VALUE;
        ColumnConfig bestCandidate = null;
        if ( CollectionUtils.isEmpty(baseColumnSet) ) {
            for ( ColumnConfig columnConfig: candidates ) {
                if ( bestCandidate == null || columnConfig.getIv() > bestCandidate.getIv() ) {
                    bestCandidate = columnConfig;
                }
            }

            LOG.info("find best variable - {} , it has highest iv - {} ", bestCandidate.getColumnName(), bestCandidate.getIv());
            return getWorkerResult(bestCandidate.getColumnNum());
        }

        ErrorConductor errorConductor = new ErrorConductor(modelConfig, columnConfigList, baseColumnSet, trainingDataSet);
        errorConductor.trainModelAndScore();

        for(ColumnConfig columnConfig: candidates) {
            if(!baseColumnSet.contains(columnConfig.getColumnNum())) {
                LOG.info("Start to test column [{}, {}]", columnConfig.getColumnNum(), columnConfig.getColumnName());

                double errorGain = errorConductor.getErrorGain(columnConfig);
                if ( errorGain > maxErrorGain ) {
                    maxErrorGain = errorGain;
                    bestCandidate = columnConfig;
                }

                LOG.info("Finish test column [{}, {}], it's error gain is - {}", columnConfig.getColumnNum(), columnConfig.getColumnName(), errorGain);
            }
        }

        LOG.info("find best variable - {} , with error - {} ", (bestCandidate == null ? "" : bestCandidate.getColumnName()), maxErrorGain);

        return ((bestCandidate == null) ? getDefaultWorkerResult() : getWorkerResult(bestCandidate.getColumnNum()));
    }

    @Override
    public VarSelWorkerResult getDefaultWorkerResult() {
        return getWorkerResult(-1);
    }

    private VarSelWorkerResult getWorkerResult(int columnId) {
        List<Integer> columnIdList = new ArrayList<Integer>();
        columnIdList.add(columnId);
        return new VarSelWorkerResult(columnIdList);
    }
}
