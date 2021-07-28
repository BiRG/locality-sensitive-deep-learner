from tensorflow.keras.layers import Layer, Dense, Flatten, Activation, Dropout, LeakyReLU, Minimum, Concatenate, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling, Constant, Zeros
from tensorflow.keras.regularizers import l1
from tensorflow.keras.constraints import non_neg, Constraint
from tensorflow.python.keras import activations
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np

class Softmin(Layer):
#Helper layer, source: user drpngx on https://github.com/tensorflow/tensorflow/issues/16028. 
#Instead of element-wise minimum y = min(a,b); we have y = z * a + (1-z) * b ; where z = sigmoid((a-b)/width)
## b is a vector of shape (1, x). a is a vector or matrix of shape (c,x). 
## Outputs a vector or matrix with the same size as input a. 
## Width: controls the spread of sigmoid curve. Smaller values approximate closer to element-wise minimum. 
	def __init__(self, 
				 width=1.,
				 **kwargs
				):
		self.width=width
		super(Softmin, self).__init__(**kwargs)
	
	def build(self, input_shape):
		super(Softmin, self).build(input_shape)
		
	def call(self, inputs):
		a=inputs[0]
		b=inputs[1]
		a_is_larger=tf.sigmoid((a-b)/self.width)
		return a_is_larger * b + (1-a_is_larger)*a 
	
	def get_config(self):
		return {'width':self.width}
	
	def compute_output_shape(self, input_shape):
		return input_shape[0]


class DenseAttention(Layer):
	def __init__(self, 
				 n_feat,
				 n_hidden,
				 out=1,
				 name_idx=0,
				 hidden_activation="sigmoid",
				 output_activation="sigmoid", 
				 kernel_initializer=VarianceScaling(),
				 bias_initializer=Zeros(),
				 **kwargs
				):
		self.n_feat=n_feat
		self.n_hidden=n_hidden
		self.out=out
		self.name_idx=name_idx
		self.kernel_initializer=kernel_initializer
		self.bias_initializer=bias_initializer
		self.hidden_activation=activations.get(hidden_activation)
		self.output_activation=activations.get(output_activation)
		super(DenseAttention, self).__init__(**kwargs)
	
	def build(self, input_shape):
		self.w1 = self.add_weight(name="attention_w1",
								  shape=(input_shape[-1], self.n_hidden),
								  initializer=self.kernel_initializer,
								  trainable=self.trainable
								 )
		self.b1 = self.add_weight(name="attention_b1",
								  shape=(self.n_hidden,),
								  initializer=self.bias_initializer,
								  trainable=self.trainable
								 )
		self.w2=self.add_weight(name="attention_w2",
								shape=(self.n_hidden, self.out),
								initializer=self.kernel_initializer,
								trainable=self.trainable
							   )
		self.b2=self.add_weight(name="attention_b2",
								shape=(self.out,),
								initializer=self.bias_initializer,
								trainable=self.trainable
							   )
		super(DenseAttention, self).build(input_shape)

	def call(self, inputs):
		hidden1=tf.math.add(tf.tensordot(inputs, self.w1, axes=[[1],[0]]), self.b1)
		act=self.hidden_activation(hidden1)
		out=tf.math.add(tf.tensordot(act, self.w2, axes=[[1],[0]]), self.b2)
		act=self.output_activation(out)
		return act
	
	def get_config(self):
		return {'n_feat':self.n_feat,
				'n_hidden':self.n_hidden,
				'out':self.out,
				'kernel_initializer':self.kernel_initializer,
				'bias_initializer':self.bias_initializer,
				'hidden_activation':self.hidden_activation,
				'output_activation':self.output_activation
			   }
	
	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.out)

class ConcatAttentions(Layer):
	def __init__(self, 
				 n_attention, 
				 n_attention_hidden, 
				 n_attention_out,
				 n_feat, 
				 n_hidden,
				 activation="sigmoid",
				 concat_activity_regularizer=None,
				 kernel_initializer=VarianceScaling(distribution="uniform"), 
				 kernel_regularizer='l1', 
				 bias_initializer=Zeros(),
				 bias_regularizer='l1', 
				 attention_kernel_initializer=VarianceScaling(distribution="uniform"),
				 attention_bias_initializer=Zeros(),
				 attention_hidden_activation="sigmoid",
				 attention_output_activation="sigmoid",
				 attention_trainable=True,
				 batch_norm_kwargs={},
				 **kwargs
			):
		self.n_attention=n_attention
		self.n_attention_hidden=n_attention_hidden
		self.n_attention_out=n_attention_out
		self.n_feat=n_feat
		self.n_hidden=n_hidden
		self.activation=activations.get(activation)
		self.concat_activity_regularizer=concat_activity_regularizer
		self.kernel_initializer=kernel_initializer
		self.kernel_regularizer=kernel_regularizer
		self.bias_initializer=bias_initializer
		self.bias_regularizer=bias_regularizer
		self.attention_kernel_initializer=attention_kernel_initializer
		self.attention_bias_initializer=attention_bias_initializer
		self.attention_hidden_activation=attention_hidden_activation
		self.attention_output_activation=attention_output_activation
		self.attention_trainable=attention_trainable
		self.batch_norm_kwargs=batch_norm_kwargs
		

		self.attention_layers=[]
		for i in range(self.n_attention):
			attention_layer=DenseAttention(n_feat=self.n_feat,
										   n_hidden=self.n_attention_hidden,
										   out=self.n_attention_out,
										   hidden_activation=self.attention_hidden_activation,
										   output_activation=self.attention_output_activation,
										   kernel_initializer=self.attention_kernel_initializer,
										   bias_initializer=self.attention_bias_initializer,
										   trainable=self.attention_trainable
										  )
			self.attention_layers.append(attention_layer)
		self.concat_layer=Concatenate(activity_regularizer=self.concat_activity_regularizer)
		#Current (v3): Use Dense layer and batch normalization
		self.dense_layer=Dense(self.n_hidden, 
			activation=None, #Batch normalization before activation
			kernel_initializer=self.kernel_initializer,
			bias_initializer=self.bias_initializer,
			kernel_regularizer=self.kernel_regularizer,
			bias_regularizer=self.bias_regularizer, 
			)
		self.batch_norm_layer=BatchNormalization(**batch_norm_kwargs)
		super(ConcatAttentions,self).__init__(**kwargs)
		
	def build(self, input_shape):
		for i in range(self.n_attention):
			if not self.attention_layers[i].built:
				self.attention_layers[i].build(input_shape)
		# self.w1 = self.add_weight(name='concat_w1',
		#                           shape=(self.n_attention*self.n_attention_out*self.n_feat, 
		#                                  self.n_hidden),
		#                           initializer=self.kernel_initializer,
		#                           regularizer=self.kernel_regularizer,
		#                           trainable=True
		#                          )
		# self.b1 = self.add_weight(name='concat_b1',
		#                           shape=(self.n_hidden,),
		#                           initializer=self.bias_initializer,
		#                           regularizer=self.bias_regularizer,
		#                           trainable=True
		#                          )     
		super(ConcatAttentions,self).build(input_shape)

	def call(self, inputs):
#         n_batch=tf.shape(inputs)[0]
#         if n_batch is None:
#             n_batch=-1
		attentions=[]
		for i in range(self.n_attention):
			attention=self.attention_layers[i](inputs)
			attentions.append(attention)
#             attentions=tf.stack([self.attention_layers[i](X)
#                 for i in range(self.n_attention)
#             ]) 
		##Previously(v1): Did not use `Concatenate` layer
#         attentions=tf.stack(attentions)
#         #n_attention by n_batch by n_attention_out
#         attentions=tf.transpose(attentions, perm=[1,0,2]) 
#         #n_batch by n_attention by n_attention_out
#         x=tf.einsum('aij,ak->aijk',attentions, inputs)  
#                     #n_batch by n_attention by n_attention_out by n_feat        
#         x=tf.reshape(x, (tf.shape(x)[0], self.n_attention*self.n_attention_out*self.n_feat)) 
#                      #n_batch by (n_attention*n_attention_out*n_feat)
		
		#New (v4): if condition for n_attention>1
		if self.n_attention>1:
			##Current (v2): Using `Concatenate` layer
			attentions=self.concat_layer(attentions)
		else:
			attentions=attentions[0]
		#n_batch by n_attention *n_attention_out
		x_out=tf.einsum('ai, ak->aik', attentions, inputs)
		#n_batch by n_attention * n_attention_out by n_feat
		x_out=tf.reshape(x_out, (tf.shape(x_out)[0], self.n_attention * self.n_attention_out * self.n_feat))
		#n_batch by n_attention * n_attention_out * n_feat

		##Previously (v2): Did not use `Dense` layer so that we can set weights trainable
		# x=tf.einsum('ai, ak -> aik', attentions, inputs)
		# x=tf.reshape(x, (tf.shape(x)[0], self.n_attention * self.n_attention_out * self.n_feat))
		# out=tf.math.add(tf.tensordot(x, self.w1, axes=[[1],[0]]), self.b1)
		
		##Current (v3): Using `Dense` layer
		hidden=self.dense_layer(x_out)        

		eps=np.finfo(np.float32).eps        
		act=self.activation(hidden+eps)

		##New (v4): Shift Batch Normalization to after non-linearinty as suggested in paper "All you need is a good init"
		act=self.batch_norm_layer(act)		
		return act

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.n_hidden)
	
	def get_config(self):
		return  {
			'n_attention': self.n_attention,
			'n_attention_hidden': self.n_attention_hidden,
			'n_attention_out': self.n_attention_out,
			'n_feat': self.n_feat,
			'n_hidden': self.n_hidden,
			'activation': self.activation,
			'concat_activity_regularizer': self.concat_activity_regularizer,
			'kernel_initializer': self.kernel_initializer,
			'kernel_regularizer': self.kernel_regularizer,
			'bias_initializer': self.bias_initializer,
			'bias_regularizer': self.bias_regularizer, 
			'attention_kernel_initializer': self.attention_kernel_initializer,
			'attention_bias_initializer': self.attention_bias_initializer,
			'attention_hidden_activation': self.attention_hidden_activation,
			'attention_output_activation': self.attention_output_activation,
			'attention_trainable': self.attention_trainable,
			'batch_norm_kwargs': self.batch_norm_kwargs
		}
		
class AttentionModel(Model):
	def __init__(self, 
				 n_attention, 
				 n_attention_hidden, 
				 n_attention_out,
				 n_feat, 
				 n_concat_hidden,
				 n_hidden1,
				 activation="sigmoid",
				 hidden_activation="sigmoid", 
				 concat_activation="sigmoid", 
				 concat_activity_regularizer=None,
				 kernel_initializer=VarianceScaling(distribution="uniform"), 
				 kernel_regularizer='l1',
				 bias_initializer=Zeros(),
				 bias_regularizer='l1', 
				 attention_initializer=VarianceScaling(distribution="uniform"),
				 attention_hidden_activation="sigmoid",
				 attention_output_activation="sigmoid", 
				 attention_trainable=True,
			 loss='binary_crossentropy',
			 **kwargs
			):
		super(AttentionModel, self).__init__(**kwargs)        
		self.n_attention=n_attention
		self.n_attention_hidden=n_attention_hidden
		self.n_attention_out=n_attention_out
		self.n_feat=n_feat
		self.n_concat_hidden=n_concat_hidden
		self.n_hidden1=n_hidden1
		self.activation=activations.get(activation)
		self.hidden_activation=hidden_activation
		self.concat_activation=concat_activation
		self.concat_activity_regularizer=concat_activity_regularizer
		self.kernel_initializer=kernel_initializer
		self.kernel_regularizer=kernel_regularizer
		self.bias_initializer=bias_initializer
		self.bias_regularizer=bias_regularizer
		self.attention_initializer=attention_initializer
		self.attention_hidden_activation=attention_hidden_activation
		self.attention_output_activation=attention_output_activation
		self.attention_trainable=attention_trainable

		self.attentions=ConcatAttentions(
				n_attention=self.n_attention, 
				n_attention_hidden=self.n_attention_hidden, 
				n_attention_out=self.n_attention_out,
				 n_feat=self.n_feat, 
				 n_hidden=self.n_concat_hidden,
				 activation=self.concat_activation,
				 concat_activity_regularizer=self.concat_activity_regularizer,
				 kernel_initializer=self.kernel_initializer, 
				 kernel_regularizer=self.kernel_regularizer,
				 bias_initializer=self.bias_initializer,
				 bias_regularizer=self.bias_regularizer,
				 attention_initializer=self.attention_initializer,
				 attention_hidden_activation=self.attention_hidden_activation,
				 attention_output_activation=self.attention_output_activation,
			attention_trainable=self.attention_trainable
		)
		self.dense1=Dense(n_hidden1, 
						  activation=self.hidden_activation
						 )#input_shape=(self.n_attention*self.n_attention_out,))
		self.output_layer=Dense(1)

	def build(self, input_shape):
		super(AttentionModel, self).build(input_shape)

	def call(self, inputs):#, training=False, y=None):
		x=self.attentions(inputs)
		hidden1=self.dense1(x)
		output=self.output_layer(hidden1)
		eps=np.finfo(np.float32).eps
		act=self.activation(output+eps)
#         if training:
#             assert y is not None, "Parameter `y` must be set if training==True" 
#             n_batch=inputs.shape[0]
#             if n_batch is None:
#                 n_batch=-1

#             with tf.GradientTape(persistent=True) as tape:
#                 x=self.attentions(inputs)
#                 hidden=self.dense1(x)
#                 output=self.output_layer(hidden)
#                 loss=self.loss(y, output)
#             grads=tape.gradient(loss, self.trainable_variables)
#             self.optimizer.apply_gradients(zip(reduced_grads, self.trainable_variables))

#         else:
#             x=self.attentions(inputs)
#             hidden=self.dense1(x)
#             output=self.output_layer(hidden)                
		return act

	def get_config(self):
		return {'n_attention': self.n_attention,
				'n_attention_out': self.n_attention_out,
				'n_feat': self.n_feat,
				'n_hidden1': self.n_hidden1,
				'activation': self.activation,                
				'hidden_activation': self.hidden_activation,
				'concat_activation': self.concat_activation,
				'kernel_initializer': self.kernel_initializer, 
				'kernel_regularizer': self.kernel_regularizer,
				'bias_initializer': self.bias_initializer,
				'bias_regularizer': self.bias_regularizer,
				'attention_initializer': self.attention_initializer,
				'attention_hidden_activation': self.attention_hidden_activation,
				'attention_output_activation': self.attention_output_activation
			   }

from tensorflow.keras.models import Model

# ##Implement custom constraint for weighting layer's feature_weights: NonNegUnitNorm
# class NonNegUnitNorm(Constraint):
#     def __init__(self, axis=0):
#         self.axis=axis
	
#     def __call__(self, w):
#         w = w * K.cast(K.greater_equal(w, 0.), K.floatx())
#         return  w / (K.epsilon() + K.sqrt(K.sum(w,
#                                                axis=self.axis,
#                                                keepdims=True)))
	
#     def get_config(self):
#         return {axis:self.axis}

class NonNegL1Norm(Constraint):
	def __init__(self, axis=0, n_feat=1):
		self.axis=axis
		self.n_feat=n_feat
		
	def __call__(self, w):
		w = w * K.cast(K.greater_equal(w, 0.), K.floatx())
		w = w / (K.epsilon() + K.sum(w))
		return w *self.n_feat

class DenseAttentionwFeatWeights(Layer):
	def __init__(self, 
				 n_feat,
				 n_hidden,
				 out=1,
				 name_idx=0,
				 hidden_activation="sigmoid",
				 output_activation="sigmoid",
				 feat_weight_trainable=True,
				 kernel_initializer=VarianceScaling(),
				 bias_initializer=Zeros(),
				 width=1., #Width of Softmin layer
				 **kwargs
				):
		self.n_feat=n_feat
		self.n_hidden=n_hidden
		self.out=out
		self.name_idx=name_idx
		self.kernel_initializer=kernel_initializer
		self.bias_initializer=bias_initializer
		self.feat_weight_trainable=feat_weight_trainable
		self.hidden_activation=activations.get(hidden_activation)
		self.output_activation=activations.get(output_activation)
		self.width=width
		self.min_layer=Softmin(width=self.width)
		super(DenseAttentionwFeatWeights, self).__init__(**kwargs)
	
	def build(self, input_shape):
		self.feat_weights = self.add_weight(name="attention_feat_weights",
											shape=(1,self.n_feat),
											initializer=Constant(value=1.),
											trainable=self.feat_weight_trainable,
											constraint=NonNegL1Norm(n_feat=self.n_feat),
										   )
		self.w1 = self.add_weight(name="attention_w1",
								  shape=(self.n_feat, self.n_hidden),
								  initializer=self.kernel_initializer,
								  trainable=self.trainable
								 )
		self.b1 = self.add_weight(name="attention_b1",
								  shape=(self.n_hidden,),
								  initializer=self.bias_initializer,
								  trainable=self.trainable
								 )
		self.w2=self.add_weight(name="attention_w2",
								shape=(self.n_hidden, self.out),
								initializer=self.kernel_initializer,
								trainable=self.trainable
							   )
		self.b2=self.add_weight(name="attention_b2",
								shape=(self.out,),
								initializer=self.bias_initializer,
								trainable=self.trainable
							   )
		super(DenseAttentionwFeatWeights, self).build(input_shape)

	def call(self, inputs):
		x=inputs[:,:self.n_feat]
		feat_weights = inputs[:,self.n_feat:]
		min_Fweight=self.min_layer([feat_weights, self.feat_weights])
		#min_Fweight=tf.math.minimum(feat_weights, self.feat_weights)
		#max_FWeight=max_Fweight/np.sum(max_Fweight, axis=1)
		#How to calculate x using two feat_weights and still maintain cartesian distances
		x1 = tf.math.multiply(min_Fweight, x)
		
		hidden1=tf.math.add(tf.tensordot(x1, self.w1, axes=[[1],[0]]), self.b1)
		act=self.hidden_activation(hidden1)
		out=tf.math.add(tf.tensordot(act, self.w2, axes=[[1],[0]]), self.b2)
		eps=np.finfo(np.float32).eps
		act=self.output_activation(out+eps)
		##How to call activity regularization manually
		
		return act
	
	def get_config(self):
		return {'n_feat':self.n_feat,
				'n_hidden':self.n_hidden,
				'out':self.out,
				'kernel_initializer':self.kernel_initializer,
				'bias_initializer': self.bias_initializer,
				'hidden_activation':self.hidden_activation,
				'output_activation':self.output_activation,
				'feat_weight_trainable':self.feat_weight_trainable,
				'activity_regularizer':self.activity_regularizer,
				'width':self.width,
			   }
	
	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.out)

class ConcatAttentionswFeatWeights(Layer):
	def __init__(self, 
				 n_attention, 
				 n_attention_hidden, 
				 n_attention_out,
				 n_feat, 
				 n_hidden,
				 activation="sigmoid",
				 concat_activity_regularizer=None,
				 kernel_initializer=VarianceScaling(distribution="uniform"), 
				 kernel_regularizer='l1',
				 bias_initializer=Zeros(),
				 bias_regularizer='l1',
				 attention_kernel_initializer=VarianceScaling(distribution="uniform"),
				 attention_bias_initializer=Zeros(),
				 attention_hidden_activation="sigmoid",
				 attention_output_activation="sigmoid", 
				 attention_trainable=True,
				 attention_feat_weight_trainable=True,
				 batch_norm_kwargs={},
				 **kwargs
			):
		self.n_attention=n_attention
		self.n_attention_hidden=n_attention_hidden
		self.n_attention_out=n_attention_out
		self.n_feat=n_feat
		self.n_hidden=n_hidden
		self.activation=activations.get(activation)
		self.concat_activity_regularizer=concat_activity_regularizer
		self.kernel_initializer=kernel_initializer
		self.kernel_regularizer=kernel_regularizer
		self.bias_initializer=bias_initializer
		self.bias_regularizer=bias_regularizer
		self.attention_kernel_initializer=attention_kernel_initializer
		self.attention_bias_initializer=attention_bias_initializer
		self.attention_hidden_activation=attention_hidden_activation
		self.attention_output_activation=attention_output_activation
		self.attention_trainable=attention_trainable
		self.attention_feat_weight_trainable=attention_feat_weight_trainable
		self.batch_norm_kwargs=batch_norm_kwargs
		
		
		self.attention_layers=[]
		for i in range(self.n_attention):
			attention_layer=DenseAttentionwFeatWeights(n_feat=self.n_feat,
										   n_hidden=self.n_attention_hidden,
										   out=self.n_attention_out,
										   hidden_activation=self.attention_hidden_activation,
										   output_activation=self.attention_output_activation,
										   feat_weight_trainable=self.attention_feat_weight_trainable,
										   kernel_initializer=self.attention_kernel_initializer,
										   bias_initializer=self.attention_bias_initializer,
										   trainable=self.attention_trainable
										  )
			self.attention_layers.append(attention_layer)
		self.concat_layer=Concatenate(activity_regularizer=self.concat_activity_regularizer)            
		#Current (v3): Use Dense layer and batch normalization
		self.dense_layer=Dense(self.n_hidden, 
			activation=None,
			kernel_initializer=self.kernel_initializer,
			bias_initializer=self.bias_initializer,
			kernel_regularizer=self.kernel_regularizer,
			bias_regularizer=self.bias_regularizer, 
			)
		self.batch_norm_layer=BatchNormalization(**batch_norm_kwargs)            
		super(ConcatAttentionswFeatWeights,self).__init__(**kwargs)
		
	def build(self, input_shape):
		for i in range(self.n_attention):
			if not self.attention_layers[i].built:
				self.attention_layers[i].build(input_shape)
		# self.w1 = self.add_weight(name='concat_w1',
		#                           shape=(self.n_attention*self.n_attention_out*self.n_feat, 
		#                                  self.n_hidden),
		#                           initializer=self.kernel_initializer,
		#                           regularizer=self.kernel_regularizer,
		#                           trainable=True
		#                          )
		# self.b1 = self.add_weight(name='concat_b1',
		#                           shape=(self.n_hidden,),
		#                           initializer=self.bias_initializer,
		#                           regularizer=self.bias_regularizer,
		#                           trainable=True
		#                          )     
		super(ConcatAttentionswFeatWeights,self).build(input_shape)

	def call(self, inputs):
		x, Fweight=inputs[:,:self.n_feat], inputs[:,self.n_feat:]
		x_weighted=tf.math.multiply(x, Fweight)
		#n_batch by n_feat
		
#         n_batch=tf.shape(inputs)[0]
#         if n_batch is None:
#             n_batch=-1
		attentions=[]
		for i in range(self.n_attention):
			attention=self.attention_layers[i](inputs)
			attentions.append(attention)
#             attentions=tf.stack([self.attention_layers[i](X)
#                 for i in range(self.n_attention)
#             ]) 

		##Previously (v1): Did not use `Concatenate` layer
		#attentions=tf.stack(attentions)
					##n_attention by n_batch by n_attention_out
		#attentions=tf.transpose(attentions, perm=[1,0,2]) 
					##n_batch by n_attention by n_attention_out
		#x=tf.einsum('aij,ak->aijk',attentions, x_weighted)  
					##n_batch by n_attention by n_attention_out by n_feat        
		#x=tf.reshape(x, (tf.shape(x)[0], self.n_attention*self.n_attention_out*self.n_feat)) 
					 ##n_batch by (n_attention*n_attention_out*n_feat)        
		
		#New (v4): if condition for n_attention>1 
		if self.n_attention>1:
		#Current (v2): Using `Concatenate` layer        	
			attentions=self.concat_layer(attentions)
		else:
			attentions=attentions[0]
		#n_batch by n_attention *n_attention_out
		x_out=tf.einsum('ai, ak->aik', attentions, x_weighted)
		#n_batch by n_attention * n_attention_out by n_feat
		x_out=tf.reshape(x_out, (tf.shape(x_out)[0], self.n_attention * self.n_attention_out * self.n_feat))
		#n_batch by n_attention * n_attention_out * n_feat
 
		##Previously (v2): Did not use `Dense` layer    
		# out=tf.math.add(tf.tensordot(x, self.w1, axes=[[1],[0]]), self.b1)

		##New (v3): Using `Dense` layer
		hidden=self.dense_layer(x_out)        

		eps=np.finfo(np.float32).eps        
		act=self.activation(hidden+eps)

		##New (v4): Shift Batch Normalization to after non-linearity as suggested in paper "All you need is a good init"        
		act=self.batch_norm_layer(act)        
		return act

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.n_hidden)
	
	def get_config(self):
		return  {
			'n_attention': self.n_attention,
			'n_attention_hidden': self.n_attention_hidden,
			'n_attention_out': self.n_attention_out,
			'n_feat': self.n_feat,
			'n_hidden': self.n_hidden,
			'activation': self.activation,
			'concat_activity_regularizer': self.concat_activity_regularizer,
			'kernel_initializer': self.kernel_initializer,
			'kernel_regularizer': self.kernel_regularizer,
			'bias_initializer': self.bias_initializer,
			'bias_regularizer': self.bias_regularizer,
			'attention_kernel_initializer': self.attention_kernel_initializer,
			'attention_bias_initializer': self.attention_bias_initializer,
			'attention_hidden_activation': self.attention_hidden_activation,
			'attention_output_activation': self.attention_output_activation,
			'attention_trainable': self.attention_trainable,
			'attention_feat_weight_trainable': self.attention_feat_weight_trainable,
			'batch_norm_kwargs': self.batch_norm_kwargs,
		}
		
class AttentionModelwFeatWeights(Model):
	def __init__(self, 
				 n_attention, 
				 n_attention_hidden, 
				 n_attention_out,
				 n_feat, 
				 n_concat_hidden,
				 n_hidden1,
				 n_hidden2=None,
				 activation="sigmoid",
				 hidden_activation="sigmoid", 
				 concat_activation="sigmoid", 
				 concat_activity_regularizer=None,                  
				 kernel_initializer=VarianceScaling(distribution="uniform"), 
				 kernel_regularizer='l1',
				 bias_initializer=Zeros(),
				 bias_regularizer='l1', 
				 attention_initializer=VarianceScaling(distribution="uniform"),
				 attention_hidden_activation="sigmoid",
				 attention_output_activation="sigmoid", 
				 attention_trainable=True,
				 attention_feat_weight_trainable=True,
			 **kwargs
			):
		super(AttentionModelwFeatWeights, self).__init__(**kwargs)        
		self.n_attention=n_attention
		self.n_attention_hidden=n_attention_hidden
		self.n_attention_out=n_attention_out
		self.n_feat=n_feat
		self.n_concat_hidden=n_concat_hidden
		self.n_hidden1=n_hidden1
		self.n_hidden2=n_hidden2
		self.activation=activations.get(activation)
		self.hidden_activation=activations.get(hidden_activation)
		self.concat_activation=activations.get(concat_activation)
		self.concat_activity_regularizer=concat_activity_regularizer
		self.kernel_initializer=kernel_initializer
		self.kernel_regularizer=kernel_regularizer
		self.bias_initializer=bias_initializer
		self.bias_regularizer=bias_regularizer
		self.attention_initializer=attention_initializer
		self.attention_hidden_activation=attention_hidden_activation
		self.attention_output_activation=attention_output_activation
		self.attention_trainable=attention_trainable
		self.attention_feat_weight_trainable=attention_feat_weight_trainable

		self.attentions=ConcatAttentionswFeatWeights(
				n_attention=self.n_attention, 
				n_attention_hidden=self.n_attention_hidden, 
				n_attention_out=self.n_attention_out,
				 n_feat=self.n_feat, 
				 n_hidden=self.n_concat_hidden,
				 activation=self.concat_activation,
				 concat_activity_regularizer=self.concat_activity_regularizer,
				 kernel_initializer=self.kernel_initializer, 
				 kernel_regularizer=self.kernel_regularizer,
				 bias_initializer=self.bias_initializer,
				 bias_regularizer=self.bias_regularizer,
				 attention_initializer=self.attention_initializer,
				 attention_hidden_activation=self.attention_hidden_activation,
				attention_output_activation=self.attention_output_activation,
			attention_trainable=self.attention_trainable,
			attention_feat_weight_trainable=self.attention_feat_weight_trainable
		)
		self.dense1=Dense(n_hidden1, 
						  activation=self.hidden_activation,
						  kernel_initializer=self.kernel_initializer,
						  kernel_regularizer=self.kernel_regularizer, 
						  bias_initializer=self.bias_initializer,
						  bias_regularizer=self.bias_regularizer
						 )#input_shape=(self.n_attention*self.n_attention_out,))
		if self.n_hidden2 is not None:
			self.dense2=Dense(n_hidden2,
							  activation=self.hidden_activation
							 )
		self.output_layer=Dense(1)

	def build(self, input_shape):
		super(AttentionModelwFeatWeights, self).build(input_shape)

	def call(self, inputs):#, training=False, y=None):
		x=self.attentions(inputs)
		hidden=self.dense1(x)
		if self.n_hidden2 is not None:
			hidden=self.dense2(hidden)
		output=self.output_layer(hidden)
		eps=np.finfo(np.float32).eps        
		act=self.activation(output+eps)
#         if training:
#             assert y is not None, "Parameter `y` must be set if training==True" 
#             n_batch=inputs.shape[0]
#             if n_batch is None:
#                 n_batch=-1

#             with tf.GradientTape(persistent=True) as tape:
#                 x=self.attentions(inputs)
#                 hidden=self.dense1(x)
#                 output=self.output_layer(hidden)
#                 loss=self.loss(y, output)
#             grads=tape.gradient(loss, self.trainable_variables)
#             self.optimizer.apply_gradients(zip(reduced_grads, self.trainable_variables))

#         else:
#             x=self.attentions(inputs)
#             hidden=self.dense1(x)
#             output=self.output_layer(hidden)                
		return act

	def get_config(self):
		return {'n_attention': self.n_attention,
				'n_attention_out': self.n_attention_out,
				'n_feat': self.n_feat,
				'n_hidden1': self.n_hidden1,
				#'n_hidden2': self.n_hidden2,
				'activation': self.activation,
				'hidden_activation': self.hidden_activation,
				'concat_activation': self.concat_activation, 
				'concat_activity_regularizer': self.concat_activity_regularizer, 
				'kernel_initializer': self.kernel_initializer, 
				'kernel_regularizer': self.kernel_regularizer,
				'bias_initializer': self.bias_initializer,
				'bias_regularizer': self.bias_regularizer,
				'attention_initializer': self.attention_initializer,
				'attention_hidden_activation': self.attention_hidden_activation,
				'attention_output_activation': self.attention_output_activation,
				'attention_trainable': self.attention_trainable,
				'attention_feat_weight_trainable': self.attention_feat_weight_trainable,
			   }

#New in v4: Fweights is learned by a gating procedure similar to ANML
