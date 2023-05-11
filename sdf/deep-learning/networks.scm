;(manage 'new 'deep-learning)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;Blocks
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define block
  (lambda (fn shape-lst)
    (list fn shape-lst)))

(define block-fn
  (lambda (ba)
    (ref ba 0)))

(define block-ls
  (lambda (ba)
    (ref ba 1)))

#|
Building a network from blocks

(define 3-layer-network
  (stack-blocks
   (list
    layer1
    layer2
    layer3)))

stack-blocks takes blocks and makes a new block whose function is combo of the input blocks functions and the shape is made by joining the individual blocks
|#

#|
block-compose returns a block that expects a tensor t and a theta, then invokes f on t and theta
Then that result is sent to g with the first j values of theta removed
|#


(define block-compose
  (lambda (f g j)
    (lambda (t)
      (lambda (theta)
	    ((g
	      ((f t) theta))
	     (drop theta j))))))

#|
(block-compose relu relu 2) is the same as 2-relu
|#

#|stack2 takes two blocks ba and bb|#

(define stack2
  (lambda (ba bb)
    (block
     (block-compose
      (block-fn ba)
      (block-fn bb)
      (len (block-ls ba)))
     (append
      (block-ls ba)
      (block-ls bb)))))

#|
bls is a list of blocks, ba is the first block in bls and rbls are the remaining blocks
|#

(define stack-blocks
  (lambda (bls)
    (stacked-blocks (drop bls 1) (ref bls 0))))

(define stacked-blocks
  (lambda (rbls ba)
    (cond
     ((null? rbls) ba)
     (else
      (stacked-blocks (drop rbls 1)
		      (stack2 ba (ref rbls 0)))))))

(define layer1
  (block relu
	 (list
	  (list 64 32)
	  (list 64))))

(define layer2
  (block relu
	 (list
	  (list 45 64)
	  (list 45))))

(define layer3
  (block relu
	 (list
	  (list 26 45)
	  (list 26))))


(stack-blocks
 (list
  layer1
  layer2
  layer3)) ;Value: (64 32) (64) (45 64) (45) (26 45) (26)

#|
makes dense layer block with relu as the block function and layer of m neurons working on tensors of length n
|#

(define dense-block
  (lambda (n m)
    (block relu
	   (list
	    (list m n)
	    (list m)))))

(define layer1
  (dense-block 32 64))

(define layer2
  (dense-block 64 45))

(define layer3
  (dense-block 45 26))

#|Example from Chapter 13 of 'The Little Learner'|#

(define iris-network
  (stack-blocks
   (list
    (dense-block 4 6)
    (dense-block 6 3))))

#|Picking weights for a network to train on initially can be a challenge, if too big by the end of the network the output becomes too large also known as exploding
If too small then then outputs become extremely tiny by the end which is known as vanishing

The bias tensor is initialized to 0.0 and the weight tensor is random scalars with a central value of 0.0 and variance of 2/n where n is length of input for that layer
'The Little Learner' Chapter 13, pg: 262
|#

(define zero-tensor
  (lambda (s)
    (build-tensor s (lambda (tidx) 0.0))))

(define zero
  (zero-tensor (list 5)))

(define t-random
  (lambda (t)
    (s:- (random 2.0) 1.0)))
  
(define random-tensor
  (lambda (s)
    (let ((t (zero-tensor s)))
      (differentiable-map t-random t))))  

(define init-shape
  (lambda (s)
    (cond
     ((= (len s) 1) (zero-tensor s))
     ((= (len s) 2)
      (random-tensor s)))))

(define iris-classifier
  (block-fn iris-network))

(define iris-theta-shapes
  (block-ls iris-network))

(define model
  (lambda (target theta)
    (lambda (t)
      ((target t) theta))))


(define init-theta
  (lambda (shapes)
    (map init-shape shapes)))

(define revs 2000)
(define learning-rate 0.0002)

