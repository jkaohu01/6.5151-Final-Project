;;; =================================================================
;;; Operator Extensions to Tensors
;;; See Interlude I and Interlude V of "The Little Learner" for
;;; more details
;;; =================================================================

;; debugging function for converting the scalar leaves of some
;; tensor differentiable from duals to numbers. This makes it
;; easier to read when printed as output because the duals are
;; not shown in their object form with links procedures
;; -----------------------------------------------------------
;; t: a tensor (all tensors are differentiables)
(define (numerize t)
  (guarantee maybe-differentiable? t numerize)
  (differentiable-map real-component t))

;; Procedure for converting a quoted list of lists into tensors.
;; This is for ease of contruction of tensors without having to do
;; as much typing as you would with manually writing out the vectors.
;; Recall that quoted lists interpret variable names as symbols. If
;; you wish to use variable names that should be evaluated, use the
;; back-quote (`) and comma (,) macro like so: `(1 2 ,var-name 4)
;; ----------------------------------------------------------------
;; quoted-list: a list produced by using the quote (') or back-quote (`)
(define (quoted->tensor quoted-list)
  (guarantee list? quoted-list quoted->tensor)
  (cond
   ((scalar? (car quoted-list)) (list->vector quoted-list))
   ((list? quoted-list) (let loop ((index 0)
                                   (result-vec (make-vector (length quoted-list)))
                                   (from-list quoted-list))
                          (if (null? from-list) 
                              result-vec
                              (begin
                                (vector-set! result-vec index (quoted->tensor (car from-list)))
                                (loop (s:+ index 1) result-vec (cdr from-list))))))
   (else
    (bad-arg-error 'quoted-list 'quoted->tensor))))

;; alias for quoted->tensor
(define q->t quoted->tensor)

#|
Tests:
(q->t '((3 2) (2 1)))
;Value: #(#(3 2) #(2 1))

(q->t '(((1 2 3) (4 5 6)) ((7 8 9) (10 11 12))))
;Value: #(#(#(1 2 3) #(4 5 6)) #(#(7 8 9) #(10 11 12)))

|#

;; Procedure for extending single input functions to tensors.
;; The input function f will be applied to the the nested
;; tensors of rank base-rank to produce a new tensor.
;; ----------------------------------------------------------
;; f: some function operating on one tensor argument of rank base-rank
;; base-rank: the numeric rank of the tensors for which f operates on
(define (ext1 f base-rank)
  (guarantee procedure? f ext1)
  (guarantee number? base-rank ext1)
  ;; tensor-object: some tensor
  (lambda (tensor-object)
    (cond
     ;; apply the operation to the tensors with correct rank
     ((of-rank? base-rank tensor-object) (f tensor-object))
     ;; descend into nested tensors to apply f
     ((tensor? tensor-object) (tmap (ext1 f base-rank) tensor-object))
     (else
      (bad-arg-error 'tensor-object 'ext1)))))

;; Produces a function taking a single tensor as input and outputs a
;; tensor whose scalars have the Scheme sqrt function applied to them
;; Note: a tensor with duals as scalars is produced as output
(define sqrt
  (ext1 sqrt-0 0))

#|
Tests:
(numerize (sqrt #(#(1 4) #(9 16))))
;Value: #(#(1 2) #(3 4))

|#

;; extending log operator to work on a single tensor
;; log has base e like in standard scheme
(define log
  (ext1 log-0 0))

#|
Tests:
(numerize (log #(1 2.718281828459045)))
;Value: #(0 1.)

|#

;; extending exp operator to work on a single tensor
(define exp
  (ext1 exp-0 0))

#|
Tests:
(numerize (exp #(0 1 2 3)))
;Value: #(1 2.718281828459045 7.38905609893065 20.085536923187668)

|#

;; Produces a function taking a single tensor as input and outputs a
;; tensor whose scalars are converted to 0.0
(define zeroes
  (ext1 (lambda (x) 0.0) 0))

#|
Tests:
(zeroes #(#(#(1 2 3) #(4 5 6)) #(#(7 8 9) #(10 11 12))))
;Value: #(#(#(0. 0. 0.) #(0. 0. 0.)) #(#(0. 0. 0.) #(0. 0. 0.)))

|#

;; Function for finding the sum of elements of rank 1 tensor
;; ----------------------------------------------------------
;; t: a rank 1 tensor
(define (sum-1 t)
  (guarantee (correct-rank-tensor? 1) t sum-1)
  (summed t (sub1 (tlen t)) 0))

;; helper for 1:sum
(define (summed t index accumulator)
  (cond
   ((zero? index) (+ (tref t 0) accumulator))
   (else
    (summed t (sub1 index) (+ (tref t index) accumulator)))))

#|
Tests:
(numerize (sum-1 #(10.0 12.0 14.0)))
;Value: 36.

(numerize (sum-1 #(1 2 3)))
;Value: 6

|#

;; Produces a function taking a single tensor as input and outputs a
;; tensor whose rank 1 tensors are folded additively into a scalar
;; Note: a tensor with duals as scalars is produced as output
(define sum
  (ext1 sum-1 1))

#|
Tests:
(numerize (sum #(#(#(1 2) #(3 4)) #(#(5 6) #(7 8)))))
;Value: #(#(3 7) #(11 15))

(numerize (sum #(#(#(1 2) #(3 4) #(5 6)) #(#(-1 1) #(-2 2) #(-3 3)))))
;Value: #(#(3 7 11) #(0 0 0))

|#

;; Takes a rank 2 tensor and flattens it into a rank 1 tensor. This means
;; that the elements of the rank 2 tensor are concatenated, in the order
;; they appear, into a single rank 1 tensor
;; -----------------------------------------------------------------------
;; t: a tensor of rank 2
(define (flatten-2 t)
  (guarantee (lambda (x) (of-rank? 2 x)) t flatten-2)
  (let ((s (shape t)))
    (let ((rows (ref s 0))
          (cols (ref s 1)))
      (make-initialized-vector (s:* rows cols)
                               (lambda (initializer-index)
                                 (tref
                                  (tref t (quotient initializer-index cols))
                                  (remainder initializer-index cols)))))))

#|
Tests:
(flatten-2 #(#(1 2) #(3 4) #(5 6)))
;Value: #(1 2 3 4 5 6)

(flatten-2 #(#(1.0 0.5) #(3.1 2.2) #(7.3 2.1)))
;Value: #(1. .5 3.1 2.2 7.3 2.1)

|#

;; Produces a function taking a single tensor as input and outputs a
;; tensor whose innermost rank 2 tensors are flattened into rank 1
;; tensors.
(define flatten
  (ext1 flatten-2 2))

#|
Tests:
(flatten #(#(#(1 2) #(3 4)) #(#(5 6) #(7 8))))
;Value: #(#(1 2 3 4) #(5 6 7 8))

(flatten #(#(#(1.0 0.5) #(3.1 2.2) #(7.3 2.1)) #(#(2.9 3.5) #(0.7 1.5) #(2.5 6.4))))
;Value: #(#(1. .5 3.1 2.2 7.3 2.1) #(2.9 3.5 .7 1.5 2.5 6.4))

|#

;; Procedure for extending binary input functions to tensors.
;; The input function f will be applied to the the nested
;; tensors such that the two binary operands to the function
;; have a specific base rank
;; ----------------------------------------------------------
;; f: some function operating on one tensor argument of rank base-rank
;; base-rank-n: the numeric rank of the tensors for which f's first operand
;;              operates on
;; base-rank-m: the numeric ronk of the tensors for which f's second operand
;;              operates on
(define (ext2 f base-rank-n base-rank-m)
  (guarantee procedure? f ext2)
  (guarantee number? base-rank-n)
  (guarantee number? base-rank-m)
  ;; tensor-t: a tensor that wil be the first operand of f
  ;; tensor-u: a tensor that will be the second operand of f
  (lambda (tensor-t tensor-u)
    (guarantee tensor? tensor-t ext2)
    (guarantee tensor? tensor-u ext2)
    (cond
     ;; reached base rank of both tensors, apply f
     ((of-ranks? base-rank-n tensor-t base-rank-m tensor-u) (f tensor-t tensor-u))
     (else
      ;; decend into tensors until reaching their base ranks
      ;; the desc function simply intelligently determines which of the two
      ;; tensors to descend into first
      (desc (ext2 f base-rank-n base-rank-m)
            base-rank-n tensor-t base-rank-m tensor-u)))))

;; Helper procedure for ext2 which descends into the tensor that still has
;; not reached its base case.
;; -----------------------------------------------------------------------
;; ext2-func: the ext2 function that lead to the calling of desc
;; base-rank-n: the base rank for the tensor-t
;; tensor-t: a tensor
;; base-rank-m: the base rank for the tensor-u
;; tensor-u: a tensor
(define (desc ext2-func base-rank-n tensor-t base-rank-m tensor-u)
  (guarantee procedure? ext2-func desc)
  (cond
   ;; reached base rank of t, descend into u to reach its base rank
   ((of-rank? base-rank-n tensor-t) (desc-u ext2-func tensor-t tensor-u))
   ;; reached base rank of u, descend into t to reach its base rank
   ((of-rank? base-rank-m tensor-u) (desc-t ext2-func tensor-t tensor-u))
   ;; descend into both tensors simultaneously when t & u's lengths match
   ((s:= (tlen tensor-t) (tlen tensor-u)) (tmap ext2-func tensor-t tensor-u))
   ;; descend into t when (rank t) > (rank u)
   ((rank> tensor-t tensor-u) (desc-t ext2-func tensor-t tensor-u))
   ;; otherwise, descend into u when (rank u) > (rank t)
   (else (desc-u ext2-func tensor-t tensor-u))))

;; Helper for desc which descends into t
(define (desc-t ext2-func tensor-t tensor-u)
  (tmap (lambda (tensor-t-element)
          (ext2-func tensor-t-element tensor-u))
        tensor-t))

;; Helper for desc which descend into u
(define (desc-u ext2-func tensor-t tensor-u)
  (tmap (lambda (tensor-u-element)
          (ext2-func tensor-t tensor-u-element))
        tensor-u))

;; extends addition operator to operate on two tensors and
;; produce one tensor. The addition operation acts on the
;; scalars of each tensor. For two equal shaped tensors,
;; the result is element-wise multiplication.
(define +
  (ext2 +-0-0 0 0))

#|
Tests:
(numerize (+ 1 2))
;Value: 3

(numerize (+ #(2) #(7)))
;Value: #(9)

(numerize (+ 1 #(1 2 3)))
;Value: #(2 3 4)

(numerize (+ #(1 2 3) 2))
;Value: #(3 4 5)

(numerize (+ #(1 2 3) #(10 20 30)))
;Value: #(11 22 33)

(numerize (+ #(#(1 2) #(3 4)) #(#(-1 -2) #(-3 -4))))
;Value: #(#(0 0) #(0 0))

(numerize (+ #(#(1 2 3) #(4 5 6)) #(#(-1 -2 -3) #(-4 -5 -6))))
;Value: #(#(0 0 0) #(0 0 0))

(numerize (+ #(#(1 2) #(3 4) #(5 6)) #(#(-1 -2) #(-3 -4) #(-5 -6))))
;Value: #(#(0 0) #(0 0) #(0 0))

(numerize (+ #(1 2 3) #(#(0 0 0) #(-1 -2 -3))))
;Value: #(#(1 2 3) #(0 0 0))

(numerize (+ #(#(-1 -2 -3) #(0 0 0)) #(1 2 3)))
;Value: #(#(0 0 0) #(1 2 3))

|#

(define -
  (ext2 --0-0 0 0))

#|
Tests:
(numerize (- #(1 2 3) 2))
;Value: #(-1 0 1)

(numerize (- #(10 20 30) #(10 20 30)))
;Value: #(0 0 0)

|#

;; extends multiplication operator to operate on two tensors
;; to produce one tensor. The multiplication operation acts
;; on the scalars of each tensor. For two equal shaped tensors,
;; the result is element-wise multiplication.
(define *
  (ext2 *-0-0 0 0))

#|
Tests:
(numerize (* 2 3))
;Value: 6

(numerize (* #(1 2 3) 2))
;Value: #(2 4 6)

(numerize (* 3 #(1 2 3)))
;Value: #(3 6 9)

(numerize (* #(1 2 3) #(10 20 30)))
;Value: #(10 40 90)

(numerize (* #(#(1 2) #(3 4)) #(#(-1 -2) #(-3 -4))))
;Value: #(#(-1 -4) #(-9 -16))

(numerize (* #(#(1 2) #(3 4) #(5 6)) #(#(-1 -2) #(-3 -4) #(-5 -6))))
;Value: #(#(-1 -4) #(-9 -16) #(-25 -36))

(numerize (* #(#(1 2 3) #(4 5 6)) #(#(-1 -2 -3) #(-4 -5 -6))))
;Value: #(#(-1 -4 -9) #(-16 -25 -36))

(numerize (* #(1 2 3) #(#(-1 -2 -3) #(10 20 30))))
;Value: #(#(-1 -4 -9) #(10 40 90))

(numerize (* #(#(1 2 3) #(10 20 30)) #(-1 -2 -3)))
;Value: #(#(-1 -4 -9) #(-10 -40 -90))

|#

(define /
  (ext2 /-0-0 0 0))

#|
Tests:
(numerize (/ #(10 20 30) #(10 10 10)))
;Value: #(1 2 3)

(numerize (/ #(10 20 30) #(2 5 3)))
;Value: #(5 4 10)

|#

(define expt
  (ext2 expt-0-0 0 0))

#|
Tests:
(numerize (expt #(0 1 2 3) 2))
;Value: #(0 1 4 9)

(numerize (expt #(0 1 2 3) #(3 3 3 3)))
;Value: #(0 1 8 27)

(numerize (expt #(0 1 2 3) #(3 3 2 3)))
;Value: #(0 1 4 27)

|#

;; extension of squaring to tensors
;; which itself uses extended * operator
(define (sqr t)
  (* t t))

#|
Tests:
(sqr #(4))
;Value: #(#(#[compound-procedure dual] 16 #[compound-procedure 55]))

(numerize (sqr #(4)))
;Value: #(16)

(numerize (sqr #(1 2 3 4)))
;Value: #(1 4 9 16)

(numerize (sqr #(#(1 2) #(3 4))))
;Value: #(#(1 4) #(9 16))

|#

;; extension of extended * operator that applies
;; extended * operator on rank 2 tensors in first argument
;; with rank 1 tensors in second argument
(define *-2-1
  (ext2 * 2 1))

#|
Tests:
(numerize (*-2-1 #(#(1 2) #(3 4)) #(10)))
;Value: #(#(#(10 20)) #(#(30 40)))

(numerize (*-2-1 #(#(1 2) #(3 4)) #(10 20)))
;Value: #(#(10 20) #(60 80))

(numerize (*-2-1 #(#(1 2) #(3 4)) #(#(10) #(20))))
;Value: #(#(#(#(10 20)) #(#(30 40))) #(#(#(20 40)) #(#(60 80))))

(numerize (*-2-1 #(#(8 1) #(7 3) #(5 4)) #(#(6 2) #(4 9) #(3 8))))
;Value: #(#(#(48 2) #(42 6) #(30 8)) #(#(32 9) #(28 27) #(20 36)) #(#(24 8) #(21 24) #(15 32)))
; see p. 192 of "Little Learner" for explanation of this exact test

|#

;; operation on two rank-1 tensors. This is the traditional
;; dot product of two vectors using the extended * operator
;; --------------------------------------------------------
;; w: a rank 1 tensor (usually representing weights)
;; t: a rank 1 tensor
(define (dot-product-1-1 w t)
  (guarantee (correct-rank-tensor? 1) w dot-product-1-1)
  (guarantee (correct-rank-tensor? 1) t dot-product-1-1)
  (sum-1 (* w t)))

#|
Tests:
(dot-product-1-1 #(1) #(1))
;Value: #(#[compound-procedure dual] 1 #[compound-procedure 50])

(numerize (dot-product-1-1 #(1) #(1)))
;Value: 1

(numerize (dot-product-1-1 #(1 0 1) #(2 100 2)))
;Value: 4

(numerize (dot-product-1-1 #(1 0 1 0 1) #(0 1 0 1 0)))
;Value: 0

|#

;; extends dot product to operate on tensors greater than rank 1
(define dot-product
  (ext2 dot-product-1-1 1 1))

#|
Tests:
(numerize (dot-product #(1 2 3) #(0 1 2)))
;Value: 8

|#
