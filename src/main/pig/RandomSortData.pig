DEFINE DuplicateTuple   ml.shifu.shifu.udf.DuplicateTuple(3);

data = LOAD '/apps/risk/det/madmen/out/huayin/ato_auto_mad/2014/{06,07}/*' USING PigStorage('\u0007', '-schema');

SPLIT data INTO positive_data IF ato_bad == 1,
    negative_data IF ato_bad == 0;

dup_pos_data = FOREACH positive_data GENERATE FLATTEN(DuplicateTuple(*));

new_data = UNION negative_data, dup_pos_data;

new_random_data = FOREACH new_data GENERATE (int)(RANDOM() * 100000000) AS seed, *;
new_order_data = ORDER new_random_data BY $1;

output_data = FOREACH new_order_data GENERATE $2..;

STORE output_data INTO 'risk/ato_auto_mad' USING PigStorage('', '-schema')