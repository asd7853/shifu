/**
 * Copyright [2012-2014] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.udf;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.WeightAmplifier;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;
import ml.shifu.shifu.core.DataSampler;
import ml.shifu.shifu.core.Normalizer;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.jexl2.Expression;
import org.apache.commons.jexl2.JexlContext;
import org.apache.commons.jexl2.JexlEngine;
import org.apache.commons.jexl2.MapContext;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.util.Utils;

/**
 * NormalizeUDF class normalize the training data
 */
/**
 * @author xiaobin
 *
 */
public class NormalizeUDF extends AbstractTrainerUDF<Tuple> {

    private List<String> negTags;
    private List<String> posTags;
    private Double cutoff;
    private NormType normType;
    private Expression weightExpr;
    private JexlContext weightContext;
    private DecimalFormat df = new DecimalFormat("#.######");
    
    public NormalizeUDF(String source, String pathModelConfig, String pathColumnConfig) throws Exception {
        super(source, pathModelConfig, pathColumnConfig);

        log.debug("Initializing NormalizeUDF ... ");

        negTags = modelConfig.getNegTags();
        log.debug("\t Negative Tags: " + negTags);
        
        posTags = modelConfig.getPosTags();
        log.debug("\t Positive Tags: " + posTags);
        
        cutoff = modelConfig.getNormalizeStdDevCutOff();
        log.debug("\t stdDevCutOff: " + cutoff);
        
        normType = modelConfig.getNormalizeType();
        log.debug("\t normType: " + normType.name());
        
        weightExpr = createExpression(modelConfig.getWeightColumnName());
        if(weightExpr != null) {
            weightContext = new MapContext();
        }

        log.debug("NormalizeUDF Initialized");
        
    }

    public Tuple exec(Tuple input) throws IOException {
        if(input == null || input.size() == 0) {
            return null;
        }

        // do data sampling. Unselected data or data with invalid tag will be filtered out.
        final String rawTag = input.get(tagColumnNum).toString();
        boolean isNotSampled = DataSampler.isNotSampled(posTags, negTags,
                modelConfig.getNormalizeSampleRate(), modelConfig.isNormalizeSampleNegOnly(), rawTag);
        if(isNotSampled) {
            return null;
        }
        
        // append tuple with tag, normalized value.
        Tuple tuple = TupleFactory.getInstance().newTuple();
        final NormType normType =  modelConfig.getNormalizeType();
        
        for(int i = 0; i < input.size(); i++) {
            ColumnConfig config = columnConfigList.get(i);
            String val = (input.get(i) == null) ? "" : input.get(i).toString();
            
            // load variables for weight calculating.
            if(weightExpr != null) {
                weightContext.set(config.getColumnName(), val);
            }

            // check tag type.
            if(tagColumnNum == i) {
                String tagType = tagTypeCheck(posTags, negTags, rawTag);
                if(tagType == null) {
                    log.error("Invalid data! The target value is not listed - " + rawTag);
                    return null;
                }
                tuple.append(tagType);
                continue;
            }

            // append normalize data.
            if(!CommonUtils.isGoodCandidate(config)) {
                tuple.append(null);
            } else {
                if(modelConfig.isOneHotEncoding() && config.isCategorical()) {
                    // TODO oneHotEncoding for categorical.
                    oneHotEncodeAppend(tuple, val, config);
                } else {
                    Double normVal = Normalizer.normalize(config, val, cutoff, normType);
                    tuple.append(df.format(normVal));
                }
            }
        }

        // append tuple with weight.
        double weight = evaluateWeight(weightExpr, weightContext);
        tuple.append(weight);

        return tuple;
    }
  
    
    /**
     * Turn categorical type value into one-hot encoded value and 
     * append it to the given tuple. If the raw value is not a valid
     * category, then a all-zero encoded sequence will be appended.
     * 
     * <p>Notice: only used for categorical type value.</p>
     * 
     * @param tuple the tuple to be appended
     * @param val the raw String value of categorical type 
     * @param config ColumnConfig 
     */
    private void oneHotEncodeAppend(Tuple tuple, String val, ColumnConfig config) {
        int index = CommonUtils.getBinNum(config, val);
        if(index == -1) {
            // invalid categorical value
            for (int i = 0; i < config.getBinCategory().size(); i++) {
                tuple.append("0");
            }
        } else {
            for (int i = 0; i < config.getBinCategory().size(); i++) {
                if(i == index) {
                    tuple.append("1");
                } else {
                    tuple.append("0");
                }
            }
        }
    }
    
    /**
     * Evaluate weight expression based on the variables context.
     * 
     * @param expr - weight evaluation expression
     * @param jc -  A JexlContext containing variables for weight expression.
     * @return The result of this evaluation
     */
    public double evaluateWeight(Expression expr, JexlContext jc) {
        double weight = 1.0d;
        if(expr != null) {
            Object result = expr.evaluate(jc);
            if(result instanceof Integer) {
                weight = ((Integer) result).doubleValue();
            } else if(result instanceof Double) {
                weight = ((Double) result).doubleValue();
            } else if(result instanceof String) {
                try {
                    weight = Double.parseDouble((String) result);
                } catch (NumberFormatException e) {
                    // Not a number, use default
                    if(System.currentTimeMillis() % 100 == 0) {
                        log.warn("Weight column type is String and value cannot be parsed with " + result
                                + ", use default 1.0d");
                    }
                    weight = 1.0d;
                }
            }
        }
        return weight;
    }
    
    /**
     * Check tag type.
     * 
     * @param posTags - positive tag list.
     * @param negTags - negtive tag list.
     * @param rawTag - raw tag string
     * @return tag type String. Return "1" for positive tag. Return "0" for negtive tag. Return null for invalid tag.
     */
    public String tagTypeCheck(List<String> posTags, List<String> negTags, String rawTag) {
        String type = null;
        if(posTags.contains(rawTag)) {
            type = "1";
        } else if(negTags.contains(rawTag)) {
            type = "0";
        }
        
        return type;
    }
    
    public Schema outputSchema(Schema input) {
        try {
            StringBuilder schemaStr = new StringBuilder();
            schemaStr.append("Normalized:Tuple(");
            for(ColumnConfig config: columnConfigList) {
                if(!config.isMeta() && config.isNumerical()) {
                    schemaStr.append(config.getColumnName() + ":float" + ",");
                } else {
                    schemaStr.append(config.getColumnName() + ":chararray" + ",");
                }
            }
            schemaStr.append("weight:float)");

            return Utils.getSchemaFromString(schemaStr.toString());
        } catch (Exception e) {
            log.error("error in outputSchema", e);
            return null;
        }
    }

    /**
     * Create expressions for multi weight settings
     * 
     * @param weightExprList
     * @return weight expression map
     */
    protected Map<Expression, Double> createExpressionMap(List<WeightAmplifier> weightExprList) {
        Map<Expression, Double> ewMap = new HashMap<Expression, Double>();

        if(CollectionUtils.isNotEmpty(weightExprList)) {
            JexlEngine jexl = new JexlEngine();

            for(WeightAmplifier we: weightExprList) {
                ewMap.put(jexl.createExpression(we.getTargetExpression()), we.getTargetWeight());
            }
        }

        return ewMap;
    }

    /**
     * Create the expression for weight setting
     * 
     * @param weightAmplifier
     * @return expression for weight amplifier
     */
    private Expression createExpression(String weightAmplifier) {
        if(StringUtils.isNotBlank(weightAmplifier)) {
            JexlEngine jexl = new JexlEngine();
            return jexl.createExpression(weightAmplifier);
        }
        return null;
    }
}
