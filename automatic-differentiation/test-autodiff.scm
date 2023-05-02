(load "~/6.5151/6.5151-Final-Project/automatic-differentiation/0-duals.scm")
(load "~/6.5151/6.5151-Final-Project/automatic-differentiation/tensors.scm")

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;Dual Tests
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;Setup

(define dual0 0)

(define k0
  end-of-chain)

(define dual1
  (dual 1 k0))

#|Should return #f|#

(dual? 1)

#|Should return #t|#

(dual? dual1)

#|Should return #t|#

(eq? (real-component dual1) 1)

#|Should return #t|#

(eq? (link-component dual1) k0)

#|Should return #t|#

(eq? (real-component dual0) 0)

#|Should return (0.0 1.0)|#

(differentiable-map (lambda (scalar-leaf) (real-component scalar-leaf))
		    (gradient-once dual1 (list dual0 dual1)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;Tensor Tests
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define scalar-tensor 3.0)

(define tensor1 (tensor 3.0 4.0 5.0))

(define tensor2 (tensor (tensor 3.0 4.0 5.0) (tensor 7.0 8.0 9.0)))

(define tensor3
  (tensor (tensor (tensor 0 1) (tensor 2 3) (tensor 4 5))
	  (tensor (tensor 6 7) (tensor 8 9) (tensor 10 11))
	  (tensor (tensor 12 13) (tensor 14 15) (tensor 16 17))
	  (tensor (tensor 18 19) (tensor 20 21) (tensor 22 23))))

#|Should return #t|#

(scalar? scalar-tensor)

#|Should return #t|#

(equal? (list->tensor (list 3.0 4.0 5.0)) tensor1)

#|Should return #t|#

(eq? (tlen tensor1) 3)

#|Should return #t|#

(equal? (tref tensor1 2) 5.0)

#|Should return #t|#

(equal? (shape tensor3) (list 4 3 2))

#|Should return #t|#

(eq? (rank tensor2) 2)

#|Should return #t|#

(eq? (size-of (shape tensor3)) 24)



#|Not working yet|#
(equal? (built-tensor
	 (lambda (idx)
	   (+ (* 6 (ref idx 0))
	      (* 2 (ref idx 1))
	      (ref idx 2)))
	 '(4 3 2)
	 '())
	                tensor3)

(build-tensor '(4 3 2)
	      (lambda (idx)
		(+ (* 6 (ref idx 0))
		   (* 2 (ref idx 1))
		   (ref idx 2))))
	           

(reshape '(24) tensor3)


(equal? (reshape '(4 1) (tensor 0 1 2 3))
	(tensor (tensor 0) (tensor 1) (tensor 2) (tensor 3)))

(equal? (reshape '(6) tensor2)
	(tensor 3.0 4.0 5.0 7.0 8.0 9.0))

(equal? (reshape '(3 2) tensor2)
	(tensor (tensor 3.0 4.0)
		(tensor 5.0 7.0)
		(tensor 8.0 9.0))))
