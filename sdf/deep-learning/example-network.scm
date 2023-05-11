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

(define revs 1000)

(define learning-rate 0.01)

(define learned-theta
  (gradient-descent ((l2-loss simple-target) simple-xs simple-ys) simple-theta))

(numerize learned-theta)

((simple-target #(2.0 7.0)) learned-theta)

(((l2-loss simple-target) simple-xs simple-ys) simple-theta)

(((l2-loss simple-target) simple-xs simple-ys) learned-theta)
