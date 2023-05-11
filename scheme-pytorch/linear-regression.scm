

(define (add-constant X)
  (vector-map
   (lambda (t)
     (let ((num-cols (get-num-cols X)))
       (let ((temp (vector-grow t (+ num-cols 1))))
         (vector-set! temp num-cols 1)
         temp)))
   X))

(define (linear-regression X y)
  (let ((X (add-constant X)))
    (m:matmul (m:matmul (invert-matrix (m:matmul (m:transpose X) X)) (m:transpose X)) y)))
