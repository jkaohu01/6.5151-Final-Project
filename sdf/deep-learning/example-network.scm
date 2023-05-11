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

(define revs 5000)

(define learning-rate 0.001)

(define learned-theta
  (gradient-descent ((l2-loss simple-target) simple-xs simple-ys) simple-theta))

(numerize learned-theta)

(numerize ((simple-target #(2.0 7.0)) learned-theta))

(numerize (((l2-loss simple-target) simple-xs simple-ys) simple-theta))

(numerize (((l2-loss simple-target) simple-xs simple-ys) learned-theta))

(define xor-net
  (stack-blocks
   (list
    (dense-block 2 3)
    (dense-block 3 3)
    (dense-block 3 1))))

(define xor-target
  (block-fn xor-net))

(define xor-theta
  (init-theta (block-ls xor-net)))

(define xor-xs
  (tensor
   (tensor 0 0)
   (tensor 0 1)
   (tensor 1 0)
   (tensor 1 1)))

(define xor-ys
  (tensor
   0
   1
   1
   0))

(define revs 2000)

(define learning-rate 0.01)

(define learned-xor-theta
  (gradient-descent ((l2-loss xor-target) xor-xs xor-ys) xor-theta))

(numerize learned-xor-theta)

(numerize ((xor-target #(0 0)) xor-theta))

(numerize ((xor-target #(0 1)) learned-xor-theta))

(numerize (((l2-loss xor-target) xor-xs xor-ys) xor-theta))

(numerize (((l2-loss xor-target) xor-xs xor-ys) learned-xor-theta))

