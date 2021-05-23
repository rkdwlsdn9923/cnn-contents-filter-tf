import tensorflow as tf

class UNet():
    def __init__(self, input_shape, output_shape):
        self.input_height = input_shape[0]
        self.input_width = input_shape[1]
        self.input_channel = input_shape[2]

        self.output_height = output_shape[0]
        self.output_width = output_shape[1]
        self.output_channel = output_shape[2]

        self.inputs = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, self.input_channel])
        self.labels = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, self.output_channel])
        self.keep_prob = tf.placeholder(tf.float32)
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.padding = 'SAME'
        self.learning_rate = 0.001
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.logits = self.model()
        self.loss = tf.reduce_mean(tf.square(self.labels - self.logits))
        self.train = self.optimizer.minimize(self.loss)
        ## W1_1 & W9_3 channel 변경 확인

    def model(self):
        
        with tf.variable_scope("layer1_1"):
            W1_1 = tf.get_variable("W1_1", shape = [3,3,3,64],initializer = tf.contrib.layers.xavier_initializer()) 
            L1_1 = tf.nn.conv2d(self.inputs, W1_1, strides=[1,1,1,1], padding = self.padding) 
            L1_1 = tf.nn.relu(L1_1) 
            L1_1 = tf.contrib.layers.batch_norm(L1_1, center=True, scale=True, is_training=self.phase)
       
        with tf.variable_scope("layer1_2"):
            W1_2 = tf.get_variable("W1_2", shape = [3,3,64,64],initializer = tf.contrib.layers.xavier_initializer()) 
            L1_2 = tf.nn.conv2d(L1_1, W1_2, strides=[1,1,1,1], padding = self.padding) 
            L1_2 = tf.nn.relu(L1_2) 
            L1_2 = tf.contrib.layers.batch_norm(L1_2, center=True, scale=True, is_training=self.phase)
        
        with tf.variable_scope("layer1_3"):
            L1_3 = tf.nn.max_pool(L1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        ################################################################################################################

        with tf.variable_scope("layer2_1"):
            W2_1 = tf.get_variable("W2_1", shape = [3,3,64,128],initializer = tf.contrib.layers.xavier_initializer()) 
            L2_1 = tf.nn.conv2d(L1_3, W2_1, strides=[1,1,1,1], padding = self.padding) 
            L2_1 = tf.nn.relu(L2_1) 
            L2_1 = tf.contrib.layers.batch_norm(L2_1, center=True, scale=True, is_training=self.phase)
       
        with tf.variable_scope("layer2_2"):
            W2_2 = tf.get_variable("W2_2", shape = [3,3,128,128],initializer = tf.contrib.layers.xavier_initializer()) 
            L2_2 = tf.nn.conv2d(L2_1, W2_2, strides=[1,1,1,1], padding = self.padding) 
            L2_2 = tf.nn.relu(L2_2) 
            L2_2 = tf.contrib.layers.batch_norm(L2_2, center=True, scale=True, is_training=self.phase)
        
        with tf.variable_scope("layer2_3"):
            L2_3 = tf.nn.max_pool(L2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        ###########################################################################################################


        with tf.variable_scope("layer3_1"):
            W3_1 = tf.get_variable("W3_1", shape = [3,3,128,256],initializer = tf.contrib.layers.xavier_initializer()) 
            L3_1 = tf.nn.conv2d(L2_3, W3_1, strides=[1,1,1,1], padding = self.padding) 
            L3_1 = tf.nn.relu(L3_1) 
            L3_1 = tf.contrib.layers.batch_norm(L3_1, center=True, scale=True, is_training=self.phase)
       
        with tf.variable_scope("layer3_2"):
            W3_2 = tf.get_variable("W3_2", shape = [3,3,256,256],initializer = tf.contrib.layers.xavier_initializer()) 
            L3_2 = tf.nn.conv2d(L3_1, W3_2, strides=[1,1,1,1], padding = self.padding) 
            L3_2 = tf.nn.relu(L3_2) 
            L3_2 = tf.contrib.layers.batch_norm(L3_2, center=True, scale=True, is_training=self.phase)
        
        with tf.variable_scope("layer3_3"):
            L3_3 = tf.nn.max_pool(L3_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        #############################################################################################################


        with tf.variable_scope("layer4_1"):
            W4_1 = tf.get_variable("W4_1", shape = [3,3,256,512],initializer = tf.contrib.layers.xavier_initializer()) 
            L4_1 = tf.nn.conv2d(L3_3, W4_1, strides=[1,1,1,1], padding = self.padding) 
            L4_1 = tf.nn.relu(L4_1) 
            L4_1 = tf.contrib.layers.batch_norm(L4_1, center=True, scale=True, is_training=self.phase)
       
        with tf.variable_scope("layer4_2"):
            W4_2 = tf.get_variable("W4_2", shape = [3,3,512,512],initializer = tf.contrib.layers.xavier_initializer()) 
            L4_2 = tf.nn.conv2d(L4_1, W4_2, strides=[1,1,1,1], padding = self.padding) 
            L4_2 = tf.nn.relu(L4_2) 
            L4_2 = tf.contrib.layers.batch_norm(L4_2, center=True, scale=True, is_training=self.phase)
            L4_2 = tf.nn.dropout(L4_2, self.keep_prob)

        with tf.variable_scope("layer4_3"):
            L4_3 = tf.nn.max_pool(L4_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        #################################################################################################################


        with tf.variable_scope("layer5_1"):
            W5_1 = tf.get_variable("W5_1", shape = [3,3,512,1024],initializer = tf.contrib.layers.xavier_initializer()) 
            L5_1 = tf.nn.conv2d(L4_3, W5_1, strides=[1,1,1,1], padding = self.padding) 
            L5_1 = tf.nn.relu(L5_1) 
            L5_1 = tf.contrib.layers.batch_norm(L5_1, center=True, scale=True, is_training=self.phase)
       
        with tf.variable_scope("layer5_2"):
            W5_2 = tf.get_variable("W5_2", shape = [3,3,1024,1024],initializer = tf.contrib.layers.xavier_initializer()) 
            L5_2 = tf.nn.conv2d(L5_1, W5_2, strides=[1,1,1,1], padding = self.padding) 
            L5_2 = tf.nn.relu(L5_2) 
            L5_2 = tf.contrib.layers.batch_norm(L5_2, center=True, scale=True, is_training=self.phase)
            L5_2 = tf.nn.dropout(L5_2, self.keep_prob)
            
        ## deconvolution 
        with tf.variable_scope("layer5_3"):
            ##W5_3 = tf.get_variable("W5_3", shape = [2,2,1024,512],initializer = tf.contrib.layers.xavier_initializer()) 
            ##L5_3 = tf.nn.conv2d_transpose(L5_2, W5_3, strides=[1,2,2,1], padding = self.padding) 
            L5_3 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=2, strides=(2,2), padding = self.padding)(L5_2)
            L5_3 = tf.nn.relu(L5_3) 
            L5_3 = tf.contrib.layers.batch_norm(L5_3, center=True, scale=True, is_training=self.phase)

        ####################################################################################################################

        with tf.variable_scope("layer6_1"):
            ## copy & crop
            L6_0 = self.copy_and_crop(L4_2, L5_3)
            W6_1 = tf.get_variable("W6_1", shape = [3,3,1024,512],initializer = tf.contrib.layers.xavier_initializer()) 
            L6_1 = tf.nn.conv2d(L6_0, W6_1, strides=[1,1,1,1], padding = self.padding) 
            L6_1 = tf.nn.relu(L6_1) 
            L6_1 = tf.contrib.layers.batch_norm(L6_1, center=True, scale=True, is_training=self.phase)
       
        with tf.variable_scope("layer6_2"):
            W6_2 = tf.get_variable("W6_2", shape = [3,3,512,512],initializer = tf.contrib.layers.xavier_initializer()) 
            L6_2 = tf.nn.conv2d(L6_1, W6_2, strides=[1,1,1,1], padding = self.padding) 
            L6_2 = tf.nn.relu(L6_2) 
            L6_2 = tf.contrib.layers.batch_norm(L6_2, center=True, scale=True, is_training=self.phase)
        ## deconvolution 
        with tf.variable_scope("layer6_3"):
            ##W6_3 = tf.get_variable("W6_3", shape = [2,2,512,256],initializer = tf.contrib.layers.xavier_initializer()) 
            ##L6_3 = tf.nn.conv2d_transpose(L6_2, W6_3, strides=[1,2,2,1], padding = self.padding) 
            L6_3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=(2,2), padding = self.padding)(L6_2)
            L6_3 = tf.nn.relu(L6_3) 
            L6_3 = tf.contrib.layers.batch_norm(L6_3, center=True, scale=True, is_training=self.phase)

        ###################################################################################################################

        with tf.variable_scope("layer7_1"):
            ## copy & crop
            L7_0 = self.copy_and_crop(L3_2, L6_3)
            W7_1 = tf.get_variable("W7_1", shape = [3,3,512,256],initializer = tf.contrib.layers.xavier_initializer()) 
            L7_1 = tf.nn.conv2d(L7_0, W7_1, strides=[1,1,1,1], padding = self.padding) 
            L7_1 = tf.nn.relu(L7_1) 
            L7_1 = tf.contrib.layers.batch_norm(L7_1, center=True, scale=True, is_training=self.phase)
       
        with tf.variable_scope("layer7_2"):
            W7_2 = tf.get_variable("W7_2", shape = [3,3,256,256],initializer = tf.contrib.layers.xavier_initializer()) 
            L7_2 = tf.nn.conv2d(L7_1, W7_2, strides=[1,1,1,1], padding = self.padding) 
            L7_2 = tf.nn.relu(L7_2) 
            L7_2 = tf.contrib.layers.batch_norm(L7_2, center=True, scale=True, is_training=self.phase)
        ## deconvolution 
        with tf.variable_scope("layer7_3"):
            ##W7_3 = tf.get_variable("W7_3", shape = [2,2,256,128],initializer = tf.contrib.layers.xavier_initializer()) 
            ##L7_3 = tf.nn.conv2d_transpose(L7_2, W7_3, strides=[1,2,2,1], padding = self.padding) 
            L7_3 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=(2,2), padding = self.padding)(L7_2)
            L7_3 = tf.nn.relu(L7_3) 
            L7_3 = tf.contrib.layers.batch_norm(L7_3, center=True, scale=True, is_training=self.phase)

        ###################################################################################################################

        with tf.variable_scope("layer8_1"):
            ## copy & crop
            L8_0 = self.copy_and_crop(L2_2, L7_3)
            W8_1 = tf.get_variable("W8_1", shape = [3,3,256,128],initializer = tf.contrib.layers.xavier_initializer()) 
            L8_1 = tf.nn.conv2d(L8_0, W8_1, strides=[1,1,1,1], padding = self.padding) 
            L8_1 = tf.nn.relu(L8_1) 
            L8_1 = tf.contrib.layers.batch_norm(L8_1, center=True, scale=True, is_training=self.phase)
       
        with tf.variable_scope("layer8_2"):
            W8_2 = tf.get_variable("W8_2", shape = [3,3,128,128],initializer = tf.contrib.layers.xavier_initializer()) 
            L8_2 = tf.nn.conv2d(L8_1, W8_2, strides=[1,1,1,1], padding = self.padding) 
            L8_2 = tf.nn.relu(L8_2) 
            L8_2 = tf.contrib.layers.batch_norm(L8_2, center=True, scale=True, is_training=self.phase)
        ## deconvolution 
        with tf.variable_scope("layer8_3"):
            ##W8_3 = tf.get_variable("W8_3", shape = [2,2,128,64],initializer = tf.contrib.layers.xavier_initializer()) 
            ##L8_3 = tf.nn.conv2d_transpose(L8_2, W8_3, strides=[1,2,2,1], padding = self.padding) 
            L8_3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=(2,2), padding = self.padding)(L8_2)
            L8_3 = tf.nn.relu(L8_3) 
            L8_3 = tf.contrib.layers.batch_norm(L8_3, center=True, scale=True, is_training=self.phase)

        ###################################################################################################################

        with tf.variable_scope("layer9_1"):
            ## copy & crop
            L9_0 = self.copy_and_crop(L1_2, L8_3)
            W9_1 = tf.get_variable("W9_1", shape = [3,3,128,64],initializer = tf.contrib.layers.xavier_initializer()) 
            L9_1 = tf.nn.conv2d(L9_0, W9_1, strides=[1,1,1,1], padding = self.padding) 
            L9_1 = tf.nn.relu(L9_1) 
            L9_1 = tf.contrib.layers.batch_norm(L9_1, center=True, scale=True, is_training=self.phase)
       
        with tf.variable_scope("layer9_2"):
            W9_2 = tf.get_variable("W9_2", shape = [3,3,64,64],initializer = tf.contrib.layers.xavier_initializer()) 
            L9_2 = tf.nn.conv2d(L9_1, W9_2, strides=[1,1,1,1], padding = self.padding) 
            L9_2 = tf.nn.relu(L9_2) 
            L9_2 = tf.contrib.layers.batch_norm(L9_2, center=True, scale=True, is_training=self.phase)
         
        with tf.variable_scope("layer9_3"):
            W9_3 = tf.get_variable("W9_3", shape = [1,1,64,3],initializer = tf.contrib.layers.xavier_initializer()) 
            L9_3 = tf.nn.conv2d(L9_2, W9_3, strides=[1,1,1,1], padding = self.padding) 
            L9_3 = tf.nn.sigmoid(L9_3) 

        ###################################################################################################################


        return L9_3

    
      
    def copy_and_crop(self, source, target):
        source_h = int(source.get_shape().as_list()[1])
        source_w = int(source.get_shape().as_list()[2])
        target_h = int(target.get_shape().as_list()[1])
        target_w = int(target.get_shape().as_list()[2])
        offset_h = int((source_h - target_h)/2)
        offset_w = int((source_w - target_w)/2)
        crop = tf.image.crop_to_bounding_box(source, offset_h, offset_w, target_h, target_w)
        copy = tf.concat([crop, target], -1)
        return copy