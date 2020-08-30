import tensorflow as tf
from transformers import AutoConfig, TFAutoModel

from .tf_layers import cbr, wave_block

def bilstm(n_units, max_text_len, num_words, emb_dim, dropout=0):
    ids = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    embedding = tf.keras.layers.Embedding(num_words+1, emb_dim, input_length=max_text_len, trainable=True)(ids)
    if dropout > 0:
        embedding = tf.keras.layers.SpatialDropout1D(dropout)(embedding)

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, return_sequences=True))(embedding)
    if dropout > 0:
        lstm = tf.keras.layers.SpatialDropout1D(dropout)(lstm)

    if dropout > 0:
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, return_sequences=False, dropout=dropout, recurrent_dropout=dropout))(lstm)
    else:
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, return_sequences=False))(lstm)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(lstm)

    model = tf.keras.models.Model(inputs=ids, outputs=out)
    
    return model

def transformer_base_model_cls_token(pretrained_model_name, max_text_len, dropout=0, bert_hidden_act='gelu', bert_hidden_dropout=False, multidrop_num=0):
    ids = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    
    mask = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    
    config = AutoConfig.from_pretrained(pretrained_model_name, \
                                        output_hidden_states=True, output_attentions=False)
    
    if bert_hidden_act != 'gelu':
        config.hidden_act = bert_hidden_act

    if bert_hidden_dropout and dropout > 0:
        config.hidden_dropout_prob = dropout

    if 'bertweet' in pretrained_model_name.lower():
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config, from_pt=True)
    else:
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config)
    
    # if config.output_hidden_states = True, obtain hidden states via basemodel(...)[-1]
    x = basemodel(ids, attention_mask=mask)[1]
    
    if multidrop_num > 0 and dropout > 0:
        multidrops = [tf.keras.layers.Dropout(dropout) for i in multidrop_num]
        denses = [tf.keras.layers.Dense(1, activation='linear') for i in multidrop_num]

    if dropout > 0:
        if multidrop_num == 0:
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        else:

            x2 = tf.keras.layers.Concatenate()([denses[i](drop_layer[i](x)) for i in range(multidrop_num)])
            x = tf.keras.layers.Dense(1, use_bias=False, activation='linear')(x2)
            x = tf.keras.layers.Activation('sigmoid')(x)

    model = tf.keras.models.Model(inputs=[ids, mask], outputs=x)
    
    return model

def transformer_base_model_mean_pooling(pretrained_model_name, max_text_len, spatial_dropout=0, dropout=0, bert_hidden_act='gelu', bert_hidden_dropout=False, multidrop_num=0):
    ids = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    
    mask = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    
    config = AutoConfig.from_pretrained(pretrained_model_name, \
                                        output_hidden_states=True, output_attentions=False)
    
    if bert_hidden_act != 'gelu':
        config.hidden_act = bert_hidden_act

    if bert_hidden_dropout and dropout > 0:
        config.hidden_dropout_prob = dropout

    if 'bertweet' in pretrained_model_name.lower():
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config, from_pt=True)
    else:
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config)
    
    # if config.output_hidden_states = True, obtain hidden states via basemodel(...)[-1]
    x = basemodel(ids, attention_mask=mask)[0]
    
    if spatial_dropout > 0:
        x = tf.keras.layers.SpatialDropout1D(spatial_dropout)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    if multidrop_num > 0 and dropout > 0:
        multidrops = [tf.keras.layers.Dropout(dropout) for i in multidrop_num]
        denses = [tf.keras.layers.Dense(1, activation='linear') for i in multidrop_num]

    if dropout > 0:
        if multidrop_num == 0:
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        else:

            x2 = tf.keras.layers.Concatenate()([denses[i](drop_layer[i](x)) for i in range(multidrop_num)])
            x = tf.keras.layers.Dense(1, use_bias=False, activation='linear')(x2)
            x = tf.keras.layers.Activation('sigmoid')(x)

    model = tf.keras.models.Model(inputs=[ids, mask], outputs=x)
    
    return model

def transformer_base_model_max_pooling(pretrained_model_name, max_text_len, spatial_dropout=0, dropout=0, bert_hidden_act='gelu', bert_hidden_dropout=False, multidrop_num=0):
    ids = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    
    mask = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    
    config = AutoConfig.from_pretrained(pretrained_model_name, \
                                        output_hidden_states=True, output_attentions=False)
    
    if bert_hidden_act != 'gelu':
        config.hidden_act = bert_hidden_act

    if bert_hidden_dropout and dropout > 0:
        config.hidden_dropout_prob = dropout

    if 'bertweet' in pretrained_model_name.lower():
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config, from_pt=True)
    else:
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config)
    
    # if config.output_hidden_states = True, obtain hidden states via basemodel(...)[-1]
    x = basemodel(ids, attention_mask=mask)[0]
    
    if spatial_dropout > 0:
        x = tf.keras.layers.SpatialDropout1D(spatial_dropout)(x)

    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    
    if multidrop_num > 0 and dropout > 0:
        multidrops = [tf.keras.layers.Dropout(dropout) for i in multidrop_num]
        denses = [tf.keras.layers.Dense(1, activation='linear') for i in multidrop_num]

    if dropout > 0:
        if multidrop_num == 0:
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        else:

            x2 = tf.keras.layers.Concatenate()([denses[i](drop_layer[i](x)) for i in range(multidrop_num)])
            x = tf.keras.layers.Dense(1, use_bias=False, activation='linear')(x2)
            x = tf.keras.layers.Activation('sigmoid')(x)

    model = tf.keras.models.Model(inputs=[ids, mask], outputs=x)
    
    return model

def transformer_with_cnn_with_meanpooling(pretrained_model_name, max_text_len, spatial_dropout=0, dropout=0, bert_hidden_act='gelu', bert_hidden_dropout=False, multidrop_num=0):
    ids = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)

    masks = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)

    config = AutoConfig.from_pretrained(pretrained_model_name, output_hidden_states = True)

    if bert_hidden_act != 'gelu':
        config.hidden_act = bert_hidden_act

    if bert_hidden_dropout and dropout > 0:
        config.hidden_dropout_prob = dropout

    if 'bertweet' in pretrained_model_name.lower():
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config, from_pt=True)
    else:
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config)

    # if config.output_hidden_states = True, obtain hidden states via basemodel(...)[-1]
    embedding = basemodel([ids, masks])

    states = tf.keras.layers.Concatenate(-1)([embedding[-1][9],embedding[-1][10],embedding[-1][11],embedding[-1][12]])

    emb = embedding[-1][0]

    if spatial_dropout > 0:
        states = tf.keras.layers.SpatialDropout1D(spatial_dropout)(states)
        emb = tf.keras.layers.SpatialDropout1D(spatial_dropout)(emb)

    x1 = wave_block(states, 128, 3, 8)
    x1 = tf.keras.layers.LayerNormalization()(x1) #BatchNormalization()(x)
    x1 = wave_block(x1, 64, 3, 4)

    x2 = wave_block(emb, 128, 3, 8)
    x2 = tf.keras.layers.LayerNormalization()(x2) #BatchNormalization()(x)
    x2 = wave_block(x2, 64, 3, 4)

    x = tf.keras.layers.Concatenate(-1)([x1, x2])

    x = tf.keras.layers.GlobalAveragePooling1D()(x1)

    if multidrop_num > 0 and dropout > 0:
        multidrops = [tf.keras.layers.Dropout(dropout) for i in multidrop_num]
        denses = [tf.keras.layers.Dense(1, activation='linear') for i in multidrop_num]

    if dropout > 0:
        if multidrop_num == 0:
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        else:
            x2 = tf.keras.layers.Concatenate()([denses[i](drop_layer[i](x)) for i in range(multidrop_num)])
            x = tf.keras.layers.Dense(1, use_bias=False, activation='linear')(x2)
            x = tf.keras.layers.Activation('sigmoid')(x)

    model = tf.keras.models.Model(inputs=[ids,masks], outputs=x)

    return model

def transformer_with_cnn_with_maxpooling(pretrained_model_name, max_text_len, spatial_dropout=0, dropout=0, bert_hidden_act='gelu', bert_hidden_dropout=False, multidrop_num=0):
    ids = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)

    masks = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)

    config = AutoConfig.from_pretrained(pretrained_model_name, output_hidden_states = True)

    if bert_hidden_act != 'gelu':
        config.hidden_act = bert_hidden_act

    if bert_hidden_dropout and dropout > 0:
        config.hidden_dropout_prob = dropout

    if 'bertweet' in pretrained_model_name.lower():
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config, from_pt=True)
    else:
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config)

    embedding = basemodel([ids, masks])

    states = tf.keras.layers.Concatenate(-1)([embedding[-1][9],embedding[-1][10],embedding[-1][11],embedding[-1][12]])

    emb = embedding[-1][0]

    if spatial_dropout > 0:
        states = tf.keras.layers.SpatialDropout1D(spatial_dropout)(states)
        emb = tf.keras.layers.SpatialDropout1D(spatial_dropout)(emb)

    x1 = wave_block(states, 128, 3, 8)
    x1 = tf.keras.layers.LayerNormalization()(x1) #BatchNormalization()(x)
    x1 = wave_block(x1, 64, 3, 4)

    x2 = wave_block(emb, 128, 3, 8)
    x2 = tf.keras.layers.LayerNormalization()(x2) #BatchNormalization()(x)
    x2 = wave_block(x2, 64, 3, 4)

    x = tf.keras.layers.Concatenate(-1)([x1, x2])

    x = tf.keras.layers.GlobalMaxPooling1D()(x1)

    if multidrop_num > 0 and dropout > 0:
        multidrops = [tf.keras.layers.Dropout(dropout) for i in multidrop_num]
        denses = [tf.keras.layers.Dense(1, activation='linear') for i in multidrop_num]

    if dropout > 0:
        if multidrop_num == 0:
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        else:
            x2 = tf.keras.layers.Concatenate()([denses[i](drop_layer[i](x)) for i in range(multidrop_num)])
            x = tf.keras.layers.Dense(1, use_bias=False, activation='linear')(x2)
            x = tf.keras.layers.Activation('sigmoid')(x)

    model = tf.keras.models.Model(inputs=[ids,masks], outputs=x)

    return model

def transformers_with_all_layers_with_meanpooling(pretrained_model_name, max_text_len, layer_nums=[], spatial_dropout=0, dropout=0, \
                                                            bert_hidden_act='gelu', bert_hidden_dropout=False, multidrop_num=0):
    ids = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    masks = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)

    config = AutoConfig.from_pretrained(pretrained_model_name, output_hidden_states = True)

    if bert_hidden_act != 'gelu':
        config.hidden_act = bert_hidden_act

    if bert_hidden_dropout and dropout > 0:
        config.hidden_dropout_prob = dropout

    if 'bertweet' in pretrained_model_name.lower():
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config, from_pt=True)
    else:
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config)

    # if config.output_hidden_states = True, obtain hidden states via basemodel(...)[-1]
    embedding = basemodel([ids, masks])

    #print (len(embedding), embedding[0].shape, embedding[1].shape)

    if layer_nums == []:
        x = tf.keras.layers.Concatenate()([embedding[-1][i] for i in range(1, len(embedding[-1]))])

    else:
        x = tf.keras.layers.Concatenate()([embedding[-1][i] for  i in layer_nums])

    if spatial_dropout > 0:
        x = tf.keras.layers.SpatialDropout1D(spatial_dropout)(x)

    #x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(states)
    #x1 = tf.keras.layers.LeakyReLU()(x1)

    if spatial_dropout > 0:
        x = tf.keras.layers.SpatialDropout1D(spatial_dropout)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    if dropout > 0:
        if multidrop_num == 0:
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        else:

            x2 = tf.keras.layers.Concatenate()([denses[i](drop_layer[i](x)) for i in range(multidrop_num)])
            x = tf.keras.layers.Dense(1, use_bias=False, activation='linear')(x2)
            x = tf.keras.layers.Activation('sigmoid')(x)

    model = tf.keras.models.Model(inputs=[ids,masks], outputs=x)

    return model

def transformers_with_all_layers_with_maxpooling(pretrained_model_name, max_text_len, layer_nums=[], spatial_dropout=0, dropout=0, \
                                                        bert_hidden_act='gelu', bert_hidden_dropout=False, multidrop_num=0):
    ids = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    masks = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)

    config = AutoConfig.from_pretrained(pretrained_model_name, output_hidden_states = True)

    if bert_hidden_act != 'gelu':
        config.hidden_act = bert_hidden_act

    if bert_hidden_dropout and dropout > 0:
        config.hidden_dropout_prob = dropout

    if 'bertweet' in pretrained_model_name.lower():
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config, from_pt=True)
    else:
        basemodel = TFAutoModel.from_pretrained(pretrained_model_name,config=config)

    # if config.output_hidden_states = True, obtain hidden states via basemodel(...)[-1]
    embedding = basemodel([ids, masks])

    #print (len(embedding), embedding[0].shape, embedding[1].shape)

    if layer_nums == []:
        x = tf.keras.layers.Concatenate()([embedding[-1][i] for i in range(1, len(embedding[-1]))])

    else:
        x = tf.keras.layers.Concatenate()([embedding[-1][i] for  i in layer_nums])

    if spatial_dropout > 0:
        x = tf.keras.layers.SpatialDropout1D(spatial_dropout)(x)

    #x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(states)
    #x1 = tf.keras.layers.LeakyReLU()(x1)

    if spatial_dropout > 0:
        x = tf.keras.layers.SpatialDropout1D(spatial_dropout)(x)

    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    if dropout > 0:
        if multidrop_num == 0:
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        else:

            x2 = tf.keras.layers.Concatenate()([denses[i](drop_layer[i](x)) for i in range(multidrop_num)])
            x = tf.keras.layers.Dense(1, use_bias=False, activation='linear')(x2)
            x = tf.keras.layers.Activation('sigmoid')(x)

    model = tf.keras.models.Model(inputs=[ids,masks], outputs=x)

    return model

