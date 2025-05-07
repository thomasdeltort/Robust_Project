import keras
import torch
import numpy as np
import keras.ops as K
#Pas compatible avec tous les env
from decomon.layers import DecomonLayer
from keras.layers import Lambda
from keras.models import Sequential
from decomon.models import clone
import torch.nn as nn
import pdb
# from deel.torchlip.functional import group_sort_2

# class gs2(nn.Module):
#     def forward(self, X):
#         return group_sort_2(X)
    
# class GS2(keras.layers.Layer):
#     def call(self, inputs):
#         # print('help')
#         n = inputs.shape[-1]//2
#         input_reshape = keras.ops.reshape(inputs, (-1, n, 2))
#         # apply min on first axis
#         input_min = K.expand_dims(K.min(input_reshape, -1), -1)
#         # apply max on first axis
#         input_max = K.expand_dims(K.max(input_reshape, -1), -1)

#         output = K.concatenate([input_min, input_max],-1)
#         # reshape output to have the same shape as input
#         output = K.reshape(output, inputs.shape)
#         return output
    
def init_W(N):
    # print(K.eye(N).shape)
    return K.eye(N)[None,:,None,:]

def affine_bound_groupsort_output_min_keras(lower, upper):

    # lower and upper shape : (batch, 2N)
    N = lower.shape[-1]//2
    var_w = init_W(N)

    lower_reshaped = K.reshape(lower, (-1,N, 2))#new shape (batch, N, 2)
    upper_reshaped = K.reshape(upper, (-1,N, 2))
    z_0 = lower_reshaped[:,:,0] - upper_reshaped[:,:,1]#(batch,N)
    z_1 = upper_reshaped[:,:,0] - lower_reshaped[:,:,1]#(batch,N)

    #Relaxation for Minimum Component: (batch, N)
    #Case0 z_1<0
    
    # b_up, b_low = 0
    mask_0 = K.sign(K.maximum(-z_1,0))#(batch,N) mask_0[i,j]=1 iff z_1[i,j]<0
    b_up_0 = 0*upper_reshaped#(batch, N, 2)
    b_low_0 = 0*lower_reshaped#(batch, N, 2)

    # W_up_0[j, i, 0, i]=1 because we select x_0
    ## Pas compris ici
    W_up_0 = K.concatenate([var_w, 0*var_w], -2) # (1, N, 2, N)
    W_low_0=W_up_0#(batch, N, 2)

    #Case1 z_0>0
    mask_1 = K.sign(K.maximum(z_0,0))#(batch,N) mask_1[i,j]=1 iff z_0[i,j]>0
    b_up_1 = 0*upper_reshaped#(batch, N, 2)
    b_low_1 = 0*lower_reshaped#(batch, N, 2)
    # W_up_0[j, i, 1, i]=1 because we select x_0
    W_up_1 = K.concatenate([0*var_w, var_w], -2) # (1, N, 2, N)
    W_low_1=W_up_1#(1, N, 2, N)

    
    #Case2 z_0<=0<=z_1
    mask_2 = 1 - mask_0 - mask_1
    # the min component receive (z_1*z_0)/(z_1-z_0) and the max component 0
    ### Pourquoi max component 0 ? ??
    
    b_low_2= K.concatenate([K.expand_dims((z_1*z_0)/(z_1-z_0), -1), K.expand_dims(0.*z_0, -1)],-1)

    # reshaping for shape compatibility
    coeff_low_2_0 = ((-z_0)/(z_1-z_0))[:,:,None,None] # (batch, N, 1, 1) # x_0 coefficient
    coeff_low_2_1 = (1 + z_0/(z_1-z_0))[:,:,None,None] # (batch, N, 1, 1) # x_1 coefficient
    W_low_2 = K.concatenate([var_w*coeff_low_2_0, var_w*coeff_low_2_1], -2) # (batch, N, 2, N)
    # print(W_low_2)
    
    b_up_2 = 0*upper_reshaped#(batch, N, 2)
    W_up_2_0 = K.concatenate([var_w, 0*var_w], -2) #case where x_0   # (batch, N, 2, N)
    W_up_2_1 = K.concatenate([0*var_w, var_w], -2) #case where x_1 #  # (batch, N, 2, N)
    
    W_up_2 = K.where(z_1[:,:,None,None]**2<z_0[:,:,None,None]**2, W_up_2_0, W_up_2_1)

    # expand masks to one more broadcastable dimension to match the bias shapes
    mask_0 = K.expand_dims(mask_0, -1)#(batch,N,1)
    mask_1 = K.expand_dims(mask_1, -1)#(batch,N,1)
    mask_2 = K.expand_dims(mask_2, -1)#(batch,N,1)
    b_low_min = mask_0*b_low_0 + mask_1*b_low_1 + mask_2*b_low_2#(batch, N, 2)
    b_up_min = mask_0*b_up_0 + mask_1*b_up_1 + mask_2*b_up_2#(batch, N, 2)

    # expand masks to one more broadcastable dimension to match the weight's shapes
    mask_0 = K.expand_dims(mask_0, -1)#(batch,N,1, 1)
    mask_1 = K.expand_dims(mask_1, -1)#(batch,N,1, 1)
    mask_2 = K.expand_dims(mask_2, -1)#(batch,N,1, 1)

    W_low_min = mask_0*W_low_0 + mask_1*W_low_1 + mask_2*W_low_2#(batch, N, 2, N)
    W_up_min = mask_0*W_up_0 + mask_1*W_up_1 + mask_2*W_up_2#(batch, N, 2)
    # pdb.set_trace()
    return (W_low_min, b_low_min),(W_up_min, b_up_min)



def affine_bound_groupsort_output_max_keras(lower, upper):

    N = lower.shape[-1]//2
    # lower and upper shape : (batch, 2N)
    lower_reshaped = K.reshape(lower, (-1,N, 2))#new shape (batch, N, 2)
    upper_reshaped = K.reshape(upper, (-1,N, 2))
    z_0 = lower_reshaped[:,:,0] - upper_reshaped[:,:,1]#(batch,N)
    z_1 = upper_reshaped[:,:,0] - lower_reshaped[:,:,1]#(batch,N)
    var_w = init_W(N)

    
    #Relaxation for Maximum Component
    #Case0 z_1<0
    mask_0 = K.sign(K.maximum(-z_1,0))#(batch,N) mask_0[i,j]=1 iff z_1[i,j]<0
    b_up_0 = 0*upper_reshaped#(batch, N, 2)
    b_low_0 = 0*lower_reshaped#(batch, N, 2)
    W_up_0 = K.concatenate([0*var_w, var_w], -2) # (1, N, 2, N)
    W_low_0=W_up_0#(1, N, 2, N)
    
    #Case1 z_0>0
    mask_1 = K.sign(K.maximum(z_0,0))#(batch,N) mask_1[i,j]=1 iff z_0[i,j]>0
    b_up_1 = 0*upper_reshaped#(batch, N, 2)
    b_low_1 = 0*lower_reshaped#(batch, N, 2)
    W_up_1 = K.concatenate([var_w, 0.*var_w], -2) #(1, N, 2, N)
    W_low_1=W_up_1#(1, N, 2, N)

    
    #Case2 z_0<=0<=z_1
    mask_2 = 1 - mask_0 - mask_1
    W_up_2 = 0*lower_reshaped#(batch, N, 2)


    ### ATTENTION MODIFICATION
    # b_up_2 = - K.concatenate([z_0[:,:,None], ((z_1*z_0)/(z_1-z_0))[:,:,None]], -1)  #(batch, N, 2) # INITIAL VERSION ### Bizarre je me demande s'il n'y a pas un - et pourquoi z_0 ?
    b_up_2 = - K.concatenate([0.* z_0[:,:,None], ((z_1*z_0)/(z_1-z_0))[:,:,None]], -1) 
    ###

    
    coeff_up_2_0 = (z_1/(z_1-z_0))[:,:,None,None] # (batch, N, 1, 1)
    coeff_up_2_1 = (1 - z_1/(z_1-z_0))[:,:,None,None] #(batch, N, 1, 1)

    W_up_2 = K.concatenate([coeff_up_2_0*var_w, coeff_up_2_1*var_w], -2) # (batch, N, 2, N)

    #####
    b_low_2 = 0*upper_reshaped#(batch, N, 2)
    W_low_2_0 = K.concatenate([0.*var_w, var_w], -2) #case where x_1 # (batch, N, 2, N)
    W_low_2_1 = K.concatenate([var_w, 0.*var_w], -2) #case where x_0 # (batch, N, 2, N)
    W_low_2 = K.where(z_1[:,:,None,None]**2<z_0[:,:,None,None]**2, W_low_2_0, W_low_2_1)

    mask_0 = K.expand_dims(mask_0, -1)#(batch,N,1)
    mask_1 = K.expand_dims(mask_1, -1)#(batch,N,1)
    mask_2 = K.expand_dims(mask_2, -1)#(batch,N,1)
    b_low_max = mask_0*b_low_0 + mask_1*b_low_1 + mask_2*b_low_2#(batch, N, 2)
    b_up_max = mask_0*b_up_0 + mask_1*b_up_1 + mask_2*b_up_2#(batch, N, 2)

    mask_0 = K.expand_dims(mask_0, -1)#(batch,N,1,1)
    mask_1 = K.expand_dims(mask_1, -1)#(batch,N,1,1)
    mask_2 = K.expand_dims(mask_2, -1)#(batch,N,1,1)
    W_up_max = mask_0*W_up_0 + mask_1*W_up_1 + mask_2*W_up_2#(batch, N, 2,N)
    W_low_max = mask_0*W_low_0 + mask_1*W_low_1 + mask_2*W_low_2#(batch, N, 2, N)
    return (W_low_max, b_low_max),(W_up_max, b_up_max)

def affine_bound_groupsort_output_keras(lower, upper):
    
    (W_low_max, b_low_max),(W_up_max, b_up_max) = affine_bound_groupsort_output_max_keras(lower, upper)
    (W_low_min, b_low_min),(W_up_min, b_up_min) = affine_bound_groupsort_output_min_keras(lower, upper)
    W_low_max = K.expand_dims(W_low_max, -1) #(batch,N,2,N, 1)
    W_up_max = K.expand_dims(W_up_max, -1) #(batch,N,2,N, 1)
    W_low_min = K.expand_dims(W_low_min, -1) #(batch,N,2,N, 1)
    W_up_min = K.expand_dims(W_up_min, -1) #(batch,N,2,N, 1)

    W_low = K.concatenate([W_low_min,W_low_max], -1)#(batch,N,2,N, 2)
    W_up = K.concatenate([W_up_min,W_up_max], -1)#(batch,N,2,N,2)
    b_low = b_low_max + b_low_min #(batch,N,2)
    b_up = b_up_max + b_up_min #(batch,N,2)
   
    b_low = K.reshape(b_low, (-1,lower.shape[-1])) #on reshape pour etre a la meme taille que GS2 (batch, 2N)
    b_up = K.reshape(b_up, (-1,lower.shape[-1]))
    W_low = K.reshape(W_low, (-1,lower.shape[-1],lower.shape[-1]))#(batch, 2N, 2N)
    W_up = K.reshape(W_up, (-1,lower.shape[-1],lower.shape[-1]))
    return (W_low, b_low), (W_up, b_up)

    
def affine_bound_sqrt_output_keras(lower, upper):
    # lower, upper.shape : (batch, N, N)
    W_low = (K.sqrt(upper) - K.sqrt(upper))/(upper - lower) #(batch, N, N)
    W_up = 1/(2*K.sqrt(lower)) #(batch, N, N)
    b_low = upper*(upper-(K.sqrt(upper) - K.sqrt(upper))/(upper - lower)) #(batch, N)
    b_up = (1/2)*K.sqrt(lower)#(batch, N)
    # Pour upper on a choisi a = lower mais ça marche pour tout a entre lower et upper avec :
    # y  = f(a) + f'(a)(x-a)
    return (W_low, b_low), (W_up, b_up)

def affine_bound_square_output_keras(lower, upper):
    mask_0 = K.sign(K.maximum(-upper,0))#(batch,N) mask_0[i,j]=1 iff upper[i,j]<0
    mask_1 = K.sign(K.maximum(lower,0))#(batch,N) mask_1[i,j]=1 iff lower[i,j]>0
    mask_2 = 1 - mask_0 - mask_1#(batch,N)

    # case lower > 0
    W_low_0 = K.square(upper)#(batch,N,N)
    W_up_0 = K.square(lower)#(batch,N,N)
    # case lower > 0
    low_1 = K.square(lower)#(batch,N)
    up_1 = K.square(upper)#(batch,N)
    # case upper*lower < 0
    W_low_2 = (K.sqrt(upper) - K.sqrt(upper))/(upper - lower) #(batch, N, N)
    W_up_2 = (K.square(upper) - K.square(upper))/(upper - lower) #(batch, N, N)
    b_low_2 = upper*(upper-(K.sqrt(upper) - K.sqrt(upper))/(upper - lower)) #(batch, N)
    b_up_2 = upper*(upper-(K.sqrt(upper) - K.sqrt(upper))/(upper - lower))#(batch, N)
    # Pour upper on a choisi a = lower mais ça marche pour tout a entre lower et upper avec :
    # y  = f(a) + f'(a)(x-a)

    W_low = mask_0*W_low_0 + mask_1*W_low_1 + mask_2*W_low_2#(batch, N, N)
    W_up = mask_0*W_up_0 + mask_1*W_up_1 + mask_2*W_up_2#(batch, N, N)

    b_low = mask_0*b_low_0 + mask_1*b_low_1 + mask_2*b_low_2#(batch, N)
    b_up = mask_0*b_up_0 + mask_1*b_up_1 + mask_2*b_up_2#(batch, N)

    return (W_low, b_low), (W_up, b_up)

def constant_ibp_bound_square_output_keras(lower, upper):
    mask_0 = K.sign(K.maximum(-upper,0))#(batch,N) mask_0[i,j]=1 iff upper[i,j]<0
    mask_1 = K.sign(K.maximum(lower,0))#(batch,N) mask_1[i,j]=1 iff lower[i,j]>0
    mask_2 = 1 - mask_0 - mask_1#(batch,N)
    
    # case lower > 0
    low_0 = K.square(upper)#(batch,N)
    up_0 = K.square(lower)#(batch,N)
    # case lower > 0
    low_1 = K.square(lower)#(batch,N)
    up_1 = K.square(upper)#(batch,N)
    # case upper*lower < 0
    low_2 = 0*lower#(batch,N)
    up_2 = K.square(K.amax(K.concatenate([lower,upper], -1), axis = -1))#(batch,N)

    low = mask_0*low_0 + mask_1*low_1 + mask_2*low_2#(batch, N)
    up = mask_0*up_0 + mask_1*up_1 + mask_2*up_2#(batch, N)
    return low, up
# class DecomonGroupSort2(DecomonLayer):
#     layer : GS2
#     def get_affine_bounds(self, lower, upper):
#         (W_low, b_low), (W_up, b_up) = affine_bound_groupsort_output_keras(lower, upper)
#         return W_low, b_low, W_up, b_up
#     def forward_ibp_propagate(self, lower, upper):
#         return self.layer(lower), self.layer(upper)

def init_W_beta(N):
    # print(K.eye(N).shape)
    return K.eye(N)[None,:,:,None]

def affine_bound_groupsort_output_min_keras_beta(lower, upper):

    # lower and upper shape : (batch, 2N)
    N = lower.shape[-1]//2
    var_w = init_W_beta(N) # (None,N,N,None)

    lower_reshaped = K.reshape(lower, (-1,N, 2))#new shape (batch, N, 2)
    upper_reshaped = K.reshape(upper, (-1,N, 2))
    z_0 = lower_reshaped[:,:,0] - upper_reshaped[:,:,1]#(batch,N)
    z_1 = upper_reshaped[:,:,0] - lower_reshaped[:,:,1]#(batch,N)

    #Relaxation for Minimum Component: (batch, N)
    #Case0 z_1<0
    
    # b_up, b_low = 0
    mask_0 = K.sign(K.maximum(-z_1,0))#(batch,N) mask_0[i,j]=1 iff z_1[i,j]<0
    b_up_0 = 0*upper_reshaped#(batch, N, 2)
    b_low_0 = 0*lower_reshaped#(batch, N, 2)

    # W_up_0[j, i, 0, i]=1 because we select x_0
    W_up_0 = K.concatenate([var_w, 0*var_w], -1) # (1, N, N, 2)
    W_low_0=W_up_0#(batch, N, N, 2)
 
    #Case1 z_0>0
    mask_1 = K.sign(K.maximum(z_0,0))#(batch,N) mask_1[i,j]=1 iff z_0[i,j]>0
    b_up_1 = 0*upper_reshaped#(batch, N, 2)
    b_low_1 = 0*lower_reshaped#(batch, N, 2)
    # W_up_0[j, i, 1, i]=1 because we select x_0
    W_up_1 = K.concatenate([0*var_w, var_w], -1) # (1, N, N, 2)
    W_low_1=W_up_1#(1, N, N, 2)

    
    #Case2 z_0<=0<=z_1
    mask_2 = 1 - mask_0 - mask_1
    # the min component receive (z_1*z_0)/(z_1-z_0) and the max component 0
    ### Pourquoi max component 0 ? ??
    
    b_low_2= K.concatenate([K.expand_dims((z_1*z_0)/(z_1-z_0), -1), K.expand_dims(0.*z_0, -1)],-1)

    # reshaping for shape compatibility
    coeff_low_2_0 = ((-z_0)/(z_1-z_0))[:,:,None,None] # (batch, N, 1, 1) # x_0 coefficient
    coeff_low_2_1 = (1 + z_0/(z_1-z_0))[:,:,None,None] # (batch, N, 1, 1) # x_1 coefficient
    W_low_2 = K.concatenate([var_w*coeff_low_2_0, var_w*coeff_low_2_1], -1) # (batch, N, N, 2)
    # print(W_low_2)
    
    b_up_2 = 0*upper_reshaped#(batch, N, 2)
    W_up_2_0 = K.concatenate([var_w, 0*var_w], -1) #case where x_0   # (batch, N, N, 2)
    W_up_2_1 = K.concatenate([0*var_w, var_w], -1) #case where x_1 #  # (batch, N, N, 2)
    
    W_up_2 = K.where(z_1[:,:,None,None]**2<z_0[:,:,None,None]**2, W_up_2_0, W_up_2_1)

    # expand masks to one more broadcastable dimension to match the bias shapes
    mask_0 = K.expand_dims(mask_0, -1)#(batch,N,1)
    mask_1 = K.expand_dims(mask_1, -1)#(batch,N,1)
    mask_2 = K.expand_dims(mask_2, -1)#(batch,N,1)
    b_low_min = mask_0*b_low_0 + mask_1*b_low_1 + mask_2*b_low_2#(batch, N, 2)
    b_up_min = mask_0*b_up_0 + mask_1*b_up_1 + mask_2*b_up_2#(batch, N, 2)

    # expand masks to one more broadcastable dimension to match the weight's shapes
    mask_0 = K.expand_dims(mask_0, -1)#(batch,N,1, 1)
    mask_1 = K.expand_dims(mask_1, -1)#(batch,N,1, 1)
    mask_2 = K.expand_dims(mask_2, -1)#(batch,N,1, 1)

    W_low_min = mask_0*W_low_0 + mask_1*W_low_1 + mask_2*W_low_2#(batch, N, N, 2)
    W_up_min = mask_0*W_up_0 + mask_1*W_up_1 + mask_2*W_up_2#(batch, N, N, 2)

    return (W_low_min, b_low_min),(W_up_min, b_up_min)

def affine_bound_groupsort_output_max_keras_beta(lower, upper):

    N = lower.shape[-1]//2
    # lower and upper shape : (batch, 2N)
    lower_reshaped = K.reshape(lower, (-1,N, 2))#new shape (batch, N, 2)
    upper_reshaped = K.reshape(upper, (-1,N, 2))
    z_0 = lower_reshaped[:,:,0] - upper_reshaped[:,:,1]#(batch,N)
    z_1 = upper_reshaped[:,:,0] - lower_reshaped[:,:,1]#(batch,N)
    var_w = init_W_beta(N)# (None,N,N,None)

    
    #Relaxation for Maximum Component
    #Case0 z_1<0 
    mask_0 = K.sign(K.maximum(-z_1,0))#(batch,N) mask_0[i,j]=1 iff z_1[i,j]<0
    b_up_0 = 0*upper_reshaped#(batch, N, 2)
    b_low_0 = 0*lower_reshaped#(batch, N, 2)
    W_up_0 = K.concatenate([0*var_w, var_w], -1) # (1, N, N, 2)
    W_low_0=W_up_0#(1, N, N, 2)
    
    #Case1 z_0>0
    mask_1 = K.sign(K.maximum(z_0,0))#(batch,N) mask_1[i,j]=1 iff z_0[i,j]>0
    b_up_1 = 0*upper_reshaped#(batch, N, 2)
    b_low_1 = 0*lower_reshaped#(batch, N, 2)
    W_up_1 = K.concatenate([var_w, 0.*var_w], -1) #(1, N, N, 2)
    W_low_1=W_up_1#(1, N, N, 2)

    
    #Case2 z_0<=0<=z_1
    mask_2 = 1 - mask_0 - mask_1
    W_up_2 = 0*lower_reshaped#(batch, N, 2)


    ### ATTENTION MODIFICATION
    # b_up_2 = - K.concatenate([z_0[:,:,None], ((z_1*z_0)/(z_1-z_0))[:,:,None]], -1)  #(batch, N, 2) # INITIAL VERSION ### Bizarre je me demande s'il n'y a pas un - et pourquoi z_0 ?
    b_up_2 = - K.concatenate([0.* z_0[:,:,None], ((z_1*z_0)/(z_1-z_0))[:,:,None]], -1) 
    ###

    
    coeff_up_2_0 = (z_1/(z_1-z_0))[:,:,None,None] # (batch, N, 1, 1)
    coeff_up_2_1 = (1 - z_1/(z_1-z_0))[:,:,None,None] #(batch, N, 1, 1)

    W_up_2 = K.concatenate([coeff_up_2_0*var_w, coeff_up_2_1*var_w], -1) # (batch, N, N, 2)

    #####
    b_low_2 = 0*upper_reshaped#(batch, N, 2)
    W_low_2_0 = K.concatenate([0.*var_w, var_w], -1) #case where x_1 # (batch, N, N, 2)
    W_low_2_1 = K.concatenate([var_w, 0.*var_w], -1) #case where x_0 # (batch, N, N, 2)
    W_low_2 = K.where(z_1[:,:,None,None]**2<z_0[:,:,None,None]**2, W_low_2_0, W_low_2_1)

    mask_0 = K.expand_dims(mask_0, -1)#(batch,N,1)
    mask_1 = K.expand_dims(mask_1, -1)#(batch,N,1)
    mask_2 = K.expand_dims(mask_2, -1)#(batch,N,1)
    b_low_max = mask_0*b_low_0 + mask_1*b_low_1 + mask_2*b_low_2#(batch, N, 2)
    b_up_max = mask_0*b_up_0 + mask_1*b_up_1 + mask_2*b_up_2#(batch, N, 2)

    mask_0 = K.expand_dims(mask_0, -1)#(batch,N,1,1)
    mask_1 = K.expand_dims(mask_1, -1)#(batch,N,1,1)
    mask_2 = K.expand_dims(mask_2, -1)#(batch,N,1,1)
    W_up_max = mask_0*W_up_0 + mask_1*W_up_1 + mask_2*W_up_2#(batch, N, N, 2)
    W_low_max = mask_0*W_low_0 + mask_1*W_low_1 + mask_2*W_low_2#(batch, N, N, 2)
    return (W_low_max, b_low_max),(W_up_max, b_up_max)

def affine_bound_groupsort_output_keras_beta(lower, upper):
    
    (W_low_max, b_low_max),(W_up_max, b_up_max) = affine_bound_groupsort_output_max_keras_beta(lower, upper)
    (W_low_min, b_low_min),(W_up_min, b_up_min) = affine_bound_groupsort_output_min_keras_beta(lower, upper)
    W_low_max = K.expand_dims(W_low_max, -2) #(batch,N,N,1, 2)
    W_up_max = K.expand_dims(W_up_max, -2) #(batch,N,N,1, 2)
    W_low_min = K.expand_dims(W_low_min, -2) #(batch,N,N,1, 2)
    W_up_min = K.expand_dims(W_up_min, -2) #(batch,N,N,1, 2)

    W_low = K.concatenate([W_low_min,W_low_max], -1)#(batch,N,N,2, 2)
    W_up = K.concatenate([W_up_min,W_up_max], -1)#(batch,N,N,2,2)
    b_low = b_low_max + b_low_min #(batch,N,2)
    b_up = b_up_max + b_up_min #(batch,N,2)
   
    b_low = K.reshape(b_low, (-1,lower.shape[-1])) #on reshape pour etre a la meme taille que GS2 (batch, 2N)
    b_up = K.reshape(b_up, (-1,lower.shape[-1]))
    W_low = K.reshape(W_low, (-1,lower.shape[-1],lower.shape[-1]))#(batch, 2N, 2N)
    W_up = K.reshape(W_up, (-1,lower.shape[-1],lower.shape[-1]))
    # pdb.set_trace()
    return (W_low, b_low), (W_up, b_up)