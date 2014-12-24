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
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.dvarsel.dataset.TrainingDataSet;
import ml.shifu.shifu.core.dvarsel.dataset.TrainingRecord;
import ml.shifu.shifu.util.CommonUtils;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 * Created on 12/4/2014.
 */
public class ErrorConductor {

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;
    private Set<Integer> workingColumnSet;
    private TrainingDataSet trainingDataSet;

    private MLDataSet trainingData;
    private List<Pair> posPairList;
    private List<Pair> negPairList;

    private double modelError = 0.0;

    public ErrorConductor(ModelConfig modelConfig,
                               List<ColumnConfig> columnConfigList,
                               Set<Integer> workingColumnSet,
                               TrainingDataSet trainingDataSet) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.workingColumnSet = workingColumnSet;
        this.trainingDataSet = trainingDataSet;
    }

    public void trainModelAndScore() {
        //1. prepare training data
        this.trainingData = new BasicMLDataSet();
        MLDataSet testingData = new BasicMLDataSet();

        this.trainingDataSet.generateValidateData(this.workingColumnSet,
                this.modelConfig.getCrossValidationRate(),
                trainingData,
                testingData);

        //2. build NNTrainer
        NNTrainer trainer = new NNTrainer(this.modelConfig, 1, false);
        trainer.setTrainSet(trainingData);
        trainer.setValidSet(testingData);
        trainer.disableModelPersistence();
        trainer.disableLogging();

        //3. train and get validation error
        try {
            trainer.train();
        } catch ( IOException e ) {
            throw new RuntimeException("Error when training model.", e);
        }

        BasicNetwork network = trainer.getNetwork();
        this.posPairList = new ArrayList<Pair>();
        this.negPairList = new ArrayList<Pair>();

        this.modelError = 0.0;

        int recordIndex = 0;
        Iterator<MLDataPair> iterator = trainingData.iterator();
        while ( iterator.hasNext() ) {
            MLDataPair record = iterator.next();
            MLData result = network.compute(record.getInput());
            double real = record.getIdeal().getData()[0];
            double modelScore = result.getData()[0];

            if ( Double.compare(real, 1.0) == 0 ) {
                modelError += Math.abs(1.0 - modelScore);
                posPairList.add(new Pair(trainingDataSet.getTrainRecord(recordIndex), record, modelScore));
            } else if ( Double.compare(real, 0.0) == 0 ) {
                modelError += Math.abs(modelScore);
                negPairList.add(new Pair(trainingDataSet.getTrainRecord(recordIndex), record, modelScore));
            }

            recordIndex ++;
        }

        this.modelError = this.modelError / (this.posPairList.size() + this.negPairList.size());
    }

    public double getErrorGain(ColumnConfig columnConfig) {
        int binNum = columnConfig.getBinLength();
        int columnIndex = this.trainingDataSet.getDataColumnIdList().indexOf(columnConfig.getColumnNum());

        double error = calPairListError(columnConfig, this.posPairList, binNum, columnIndex);
        error += calPairListError(columnConfig, this.negPairList, binNum, columnIndex);

        return this.modelError - error / (this.posPairList.size() + this.negPairList.size());
    }

    private double calPairListError(ColumnConfig columnConfig, List<Pair> pairList, int binCount, int columnIndex) {
        double[] posErrorArr = new double[binCount];
        double[] negErrorArr = new double[binCount];

        double errorForMissingValue = 0.0;
        for ( Pair pair : pairList ) {
            String rawText = pair.getTrainingRecord().getRaw()[columnIndex];
            int binIndex = CommonUtils.getBinNum(columnConfig, rawText);
            if ( binIndex < 0 ) {
                errorForMissingValue += Math.max(Math.abs(1.0 - pair.getModelScore()), Math.abs(pair.getModelScore()));
            } else {
                posErrorArr[binIndex] += Math.abs(1.0 - pair.getModelScore());
                negErrorArr[binIndex] += Math.abs(pair.getModelScore());
            }
        }

        double error = errorForMissingValue;
        for ( int i = 0; i < binCount; i ++ ) {
            error += Math.min(posErrorArr[i], negErrorArr[i]);
        }

        return error;
    }

    public static class Pair {
        private TrainingRecord trainingRecord;
        private MLDataPair mlDataPair;
        private double modelScore;

        public Pair(TrainingRecord trainingRecord, MLDataPair mlDataPair, double modelScore) {
            this.trainingRecord = trainingRecord;
            this.mlDataPair = mlDataPair;
            this.modelScore = modelScore;
        }

        public TrainingRecord getTrainingRecord() {
            return trainingRecord;
        }

        public void setTrainingRecord(TrainingRecord trainingRecord) {
            this.trainingRecord = trainingRecord;
        }

        public MLDataPair getMlDataPair() {
            return mlDataPair;
        }

        public void setMlDataPair(MLDataPair mlDataPair) {
            this.mlDataPair = mlDataPair;
        }

        public double getModelScore() {
            return modelScore;
        }

        public void setModelScore(double modelScore) {
            this.modelScore = modelScore;
        }
    }
}
