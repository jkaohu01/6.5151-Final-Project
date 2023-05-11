

;; (define (same-size matrices)
;;   (let ((row-len (get-num-rows (car matrices)))
;;         (col-len (get-num-cols (car matrices))))
;;     (if (any (lambda (m)
;;                (not (and
;;                      (= (get-num-rows m) row-len)
;;                      (= (get-num-cols m) col-len))))
;;              matrices)
;;         (error "Matrix dimension mismatch: " matrices))))


;;; Tensor operations

(define (tensor-element-wise operation)
  (lambda (tensors)
    (apply vector-map operation tensors)))

(define (t:+ . tensors)
  ((tensor-element-wise +) tensors))

(define (t:* . tensors)
  ((tensor-element-wise *) tensors))

(define (t:dot . tensors)
  (apply + (vector->list (apply t:* tensors))))
