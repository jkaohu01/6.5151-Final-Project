;;; Batching for Neural Nets

;; Generates a random sample of s numbers from
;; 0 to n - 1
(define samples
  (lambda (n s)
    (sampled n s (list))))

;; helper for samples
(define sampled
  (lambda (n i a)
    (cond
     ((zero? i) a)
     (else
      (sampled n (sub1 i)
               (cons (random n) a))))))

#|
Tests:
;; This is non-deterministic so results
;; will vary every run but here is one result:
(samples 20 3)
;Value: (12 9 4)

|#

;; random sampled batch generation
(define sampling-obj
  (lambda (predict xs ys)
    (let ((n (tlen xs)))
      (lambda (theta)
        (let ((batch-indices (samples n batch-size)))
          ((predict (trefs xs batch-indices) (trefs ys batch-indices)) theta))))))

#|
Tests:
(begin 
  (define revs 1000)
  (define learning-rate 0.01)
  (define batch-size 4)
  (define theta (list 0.0 0.0)))

;; this is non-deterministic so results will vary
;; but it should be around ;Value: (1.05 1.87e-6)
(numerize (gradient-descent
            (sampling-obj (l2-loss line) line-xs line-ys) theta))
;Value: (1.0796432083376486 -1.1722233111151342e-2)

|#


