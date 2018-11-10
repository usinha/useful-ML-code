def get_model(num_filters, top_k):

    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, 2 * num_filters * top_k))

    inp = Input(shape=(maxlen, ))
    layer = Embedding(max_features, embed_size, weights=[embedding_matrix])
    x = layer(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(num_filters, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    k_max = Lambda(_top_k)(x)
    conc = concatenate([avg_pool, k_max])
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
