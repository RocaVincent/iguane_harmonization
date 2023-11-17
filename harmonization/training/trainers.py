from tensorflow import function as tf_function, where as tf_where, GradientTape
from tensorflow.math import reduce_mean, square, abs as tf_abs

LAMBDA_CYC = 30

class Discriminator_trainer:
  
    def __init__(self, discriminator, generator, optimizer):
        self.discriminator = discriminator
        self.generator = generator
        self.optimizer = optimizer
    
    @tf_function(jit_compile=True)
    def train(self, images1, images2):
        mask = images2>-1
        images1 = tf_where(images1>-1, images1, 0)
        images2 = tf_where(mask, images2, 0)
        fake_images = tf_where(mask, self.generator(images2, training=False), 0)
        with GradientTape() as tape:
            disc_real = self.discriminator(images1, training=True)
            disc_fakes = self.discriminator(fake_images, training=True)
            disc_loss = reduce_mean(square(disc_real-1)) + reduce_mean(square(disc_fakes))
            disc_loss_scaled = self.optimizer.get_scaled_loss(disc_loss)
        grads = tape.gradient(disc_loss_scaled, self.discriminator.trainable_weights)
        grads = self.optimizer.get_unscaled_gradients(grads)
        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        return disc_loss
    
    
class Generator_trainer:
  
    def __init__(self, genFwd, genBwd, discFwd, discBwd, optimizerFwd, optimizerBwd):
        self.genFwd = genFwd
        self.genBwd = genBwd
        self.discFwd = discFwd
        self.discBwd = discBwd
        self.optimizerFwd = optimizerFwd
        self.optimizerBwd = optimizerBwd
        
    @tf_function(jit_compile=True)
    def train(self, batchSrc, batchRef):
        maskSrc = batchSrc>-1
        maskRef = batchRef>-1
        batchSrc = tf_where(maskSrc, batchSrc, 0)
        batchRef = tf_where(maskRef, batchRef, 0)
        with GradientTape(persistent=True) as tape:
            fakesRef = tf_where(maskSrc, self.genFwd(batchSrc, training=True), 0)
            fakesSrc = tf_where(maskRef, self.genBwd(batchRef, training=True), 0)
            
            # adversarial losses
            disc_fakesRef = self.discFwd(fakesRef, training=False)
            disc_fakesSrc = self.discBwd(fakesSrc, training=False)
            genFwd_adv_loss = reduce_mean(square(disc_fakesRef-1))
            genBwd_adv_loss = reduce_mean(square(disc_fakesSrc-1))
            
            # reconstruction losses
            cycledRef = tf_where(maskRef, self.genFwd(fakesSrc, training=True), 0)
            cycledSrc = tf_where(maskSrc, self.genBwd(fakesRef, training=True), 0)
            cycle_loss_refSref = reduce_mean(tf_abs(batchRef-cycledRef))
            cycle_loss_srcRsrc = reduce_mean(tf_abs(batchSrc-cycledSrc))
            sum_cycle_loss = cycle_loss_refSref + cycle_loss_srcRsrc
            
            # identity losses for the generators
            idRef = tf_where(maskRef, self.genFwd(batchRef, training=True), 0)
            idSrc = tf_where(maskSrc, self.genBwd(batchSrc, training=True), 0)
            genFwd_idLoss = reduce_mean(tf_abs(batchRef-idRef))
            genBwd_idLoss = reduce_mean(tf_abs(batchSrc-idSrc))
            
            # final losses
            genFwd_loss = genFwd_adv_loss + LAMBDA_CYC*sum_cycle_loss + 0.5*LAMBDA_CYC*genFwd_idLoss
            genFwd_loss = self.optimizerFwd.get_scaled_loss(genFwd_loss) # genUnivOptimizer var globale
            genBwd_loss = genBwd_adv_loss + LAMBDA_CYC*sum_cycle_loss + 0.5*LAMBDA_CYC*genBwd_idLoss
            genBwd_loss = self.optimizerBwd.get_scaled_loss(genBwd_loss)
            
        grads_genFwd = tape.gradient(genFwd_loss, self.genFwd.trainable_weights)
        grads_genFwd = self.optimizerFwd.get_unscaled_gradients(grads_genFwd)
        self.optimizerFwd.apply_gradients(zip(grads_genFwd, self.genFwd.trainable_weights))

        grads_genBwd = tape.gradient(genBwd_loss, self.genBwd.trainable_weights)
        grads_genBwd = self.optimizerBwd.get_unscaled_gradients(grads_genBwd)
        self.optimizerBwd.apply_gradients(zip(grads_genBwd, self.genBwd.trainable_weights))

        return genFwd_adv_loss, genBwd_adv_loss, cycle_loss_refSref, cycle_loss_srcRsrc, genFwd_idLoss, genBwd_idLoss