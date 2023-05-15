;;;=============================================
;;; Simple Network Test
;;;=============================================

(define simple-network
  (stack-blocks
   (list
    (dense-block 2 2)
    (dense-block 2 1))))

(define simple-target
  (block-fn simple-network))

(define simple-theta
  (init-theta (block-ls simple-network)))

(define simple-xs #(2.0 7.0))

(define simple-ys #(10.0))

(define revs 2000)

(define learning-rate 0.001)

(define learned-theta
  (gradient-descent ((l2-loss simple-target) simple-xs simple-ys) simple-theta))

(numerize learned-theta)

(numerize ((simple-target #(2.0 7.0)) learned-theta))

(numerize (((l2-loss simple-target) simple-xs simple-ys) simple-theta))

(numerize (((l2-loss simple-target) simple-xs simple-ys) learned-theta))


#|
Due to the random nature of parameter initialization and RELU, sometimes our neural network always
outputs a zero because the weights begin to vanish. This is bad because the gradient of an input with
respect to a zero weight is zero so no update can be made to the parameters during gradient descent.
As a result, it is often the case that the network will not learn anything. However, there do exist 
some good initializations that we have randomly stumbled across which result in proper gradient 
descent. Below is one such initialization and test. If we input this initialization as simple-theta 
above (instead of randomly generating a simple-theta), then the output is deterministic and we can 
verify that the neural network works.

Working tests (using the code above before this long-form comment):
1). First load everything above just before the (define learned-theta ...) statement. Do not load the
    learned-theta statment because this will run the gradient descent with the randomly initialized simple-theta.

2.) Load the following line to initalize with a good starting theta of parameters to the network: 
(define simple-theta (list #(#(-.7270086004699653 .1780248633836925) #(-4.6073211153979665e-3 -.26850093821303633)) #(0. 0.) #(#(.7095944262979634 .7098008729247383)) #(0.)))
;Value: simple-theta

3.) Load the (define learned-theta ...) statement to run gradient descent.

4.) Run the lines following the definition of learned-theta. Note that the 'numerize' procedure simply unpackages
    data from the dual data structure to make it easier to read numerically. Otherwise, we would see the dual in
    its vector form which is not easy to read. To see the outputs in their original form, simply unwrap the
    statements from the 'numerize' procedure. The output should be:

;;; The learned parameters
(numerize learned-theta)
;Value: (#(#(-.7270086004699653 .1780248633836925) #(-4.6073211153979665e-3 -.26850093821303633)) #(0. 0.) #(#(.7095944262979634 .7098008729247383)) #(9.817575747762492))

;;; The prediction made by the network with input #(2.0 7.0) and learned parameters
(numerize ((simple-target #(2.0 7.0)) learned-theta))
;Value: #(9.817575747762492)

;;; The loss of the network with the data and original parameters
(numerize (((l2-loss simple-target) simple-xs simple-ys) simple-theta))
;Value: 100.

;;; The loss of the network with the data and learned parameters
(numerize (((l2-loss simple-target) simple-xs simple-ys) learned-theta))
;Value: .03327860780441407

|#

;;;=============================================
;;; Training Network to Learn XOR function
;;;=============================================
(define xor-net
  (stack-blocks
   (list
    (dense-block 2 3)
    (dense-block 3 3)
    (dense-block 3 2)
    (dense-block 2 1))))

(define xor-target
  (block-fn xor-net))

(define xor-init-theta
  (init-theta (block-ls xor-net)))

(define xor-xs
  (tensor
   (tensor 0 0)
   (tensor 0 1)
   (tensor 1 0)
   (tensor 1 1)))

(define xor-ys
  (tensor
   (tensor 0)
   (tensor 1)
   (tensor 1)
   (tensor 0)))

(define revs 3000)

(define learning-rate 0.01)

(define xor-learned-theta
  (gradient-descent ((l2-loss xor-target) xor-xs xor-ys) xor-init-theta))

(numerize xor-learned-theta) 

(numerize (((l2-loss xor-target) xor-xs xor-ys) xor-learned-theta))

(numerize (((l2-loss xor-target) xor-xs xor-ys) xor-init-theta))

(numerize ((xor-target (tensor 0 0)) xor-learned-theta))

(numerize ((xor-target (tensor 0 1)) xor-learned-theta))

(numerize ((xor-target (tensor 1 0)) xor-learned-theta)) 

(numerize ((xor-target (tensor 1 1)) xor-learned-theta))

(numerize ((xor-target (tensor 0 0)) xor-init-theta))

(numerize ((xor-target (tensor 0 1)) xor-init-theta))

(numerize ((xor-target (tensor 1 0)) xor-init-theta))

(numerize ((xor-target (tensor 1 1)) xor-init-theta))

#|

Working test (using XOR data above):

1.) Load up everything above for the XOR test just before (define xor-learned-theta ...). Do not load the
    the (define xor-learned-theta ...) statement because it will run gradient descent with the randomly
    initialized theta of parameters.

2.)  Load the following statement to load up good theta of initial parameters for the neural net.
(define xor-init-theta (list #(#(.3091305266346802 .8193797811543895) #(.9792470858329634 -.7702557155741926) #(-.9973846113143535 .5282477445421658)) #(0. 0. 0.) #(#(.03127101299801138 -.813577109674289 -.7903086616696253) #(.39599060192981583 -.04008078339802201 1.7305814452608148e-2) #(.4853418511734251 -.2464282891971563 -.41453505683744474)) #(0. 0. 0.) #(#(.5591518136909674 -.5409519499166167 .12626613491850125) #(-.35411303048271825 -.8339547994294242 .5202475673864293)) #(0. 0.) #(#(-.6336460856951815 .9020004652797988)) #(0.)))

3.) Load (define xor-learned-theta ...) to run gradient descent.

4.) Run the remaining statements which should produce the following outputs:

;;; the learned parameters for XOR 
(numerize xor-learned-theta)
;Value: (#(#(.39959979407977597 .815612954127043) #(.9337019545229296 -.8738400556287308) #(-.9973846113143535 .7772018956657111)) #(.07275526607257603 -.05152159805801597 -3.3901605753862944e-18) #(#(.16143587057854544 -.8044123383299456 -.9381561526053833) #(.44534877184671084 .2642106980130139 .14857818936206582) #(.4828038977556321 -.3051678201640455 -.4676811005054953)) #(.5966451603128108 -.03240146839080241 .0742357527051759) #(#(.965533919676834 -.6499769643434286 .3136897699654937) #(-.35411303048271825 -.8339547994294242 .5202475673864293)) #(.2567922364476613 -7.216003722238392e-4) #(#(-1.1382784020706558 .9020004652797988)) #(.9999999999999984))

;;; loss of the network after learning theta via gradient descent (the individual losses for every input)
(numerize (((l2-loss xor-target) xor-xs xor-ys) xor-learned-theta))
;Value: #(6.039716305598372e-31 2.4158865222393487e-30 2.4158865222393487e-30 0.)

;;; initial loss of the network with starting theta (the individual losses for every input)
(numerize (((l2-loss xor-target) xor-xs xor-ys) xor-init-theta))
;Value: #(0. 6.170728545916925e-3 1. 1.0821084725126977)

;;; the output of the network with learned parameters and input (tensor 0 0)
(numerize ((xor-target (tensor 0 0)) xor-learned-theta))
;Value: #(7.771561172376096e-16)

;;; the output of the network with learned parameters and input (tensor 0 1)
(numerize ((xor-target (tensor 0 1)) xor-learned-theta))
;Value: #(.9999999999999984)

;;; the output of the network with learned parameters and input (tensor 1 0)
(numerize ((xor-target (tensor 1 0)) xor-learned-theta))
;Value: #(.9999999999999984)

;;; the output of the network with learned parameters and input (tensor 1 1)
(numerize ((xor-target (tensor 1 1)) xor-learned-theta))
;Value: #(0.)

;;; the output of the network with initial parameters and input (tensor 0 0)
(numerize ((xor-target (tensor 0 0)) xor-init-theta))
;Value: #(0.)

;;; the output of the network with initial parameters and input (tensor 0 1)
(numerize ((xor-target (tensor 0 1)) xor-init-theta))
;Value: #(0.)

;;; the output of the network with initial parameters and input (tensor 1 0)
(numerize ((xor-target (tensor 1 0)) xor-init-theta))
;Value: #(0.)

;;; the output of the network with initial parameters and input (tensor 1 1)
(numerize ((xor-target (tensor 1 1)) xor-init-theta))
;Value: #(0.)

;;; From the above outputs we see that the network learns the XOR classification.
;;; Initially, it classifies all of our data as 0 but aftwerwards it begins to
;;; classify that 1 should be outputted when the values are distinct and 0 otherwise.
;;; It does not begin to classify perfectly as the values are still a little off but
;;; we can tell that they are off by very small factors.
|#
