;;; Matrix operations

(define (get-num-rows matrix)
  (vector-length matrix))

(define (get-num-cols matrix)
  (vector-length (vector-ref matrix 0)))

(define (element-wise operation)
  (lambda (matrices)
    (same-size matrices)
    (apply vector-map operation matrices)))

(define (m:+ . matrices)
  ((element-wise t:+) matrices))

(define (m:transpose matrix)
  (vector-map
   (lambda (column)
     column)
   (apply vector-map
          vector
          (vector->list matrix))))

(define (m:matmul m1 m2)
  (assert (= (get-num-cols m1) (get-num-rows m2)))
  (vector-map
   (lambda (row)
     (vector-map
      (lambda (column)
        (t:dot row column))
      (apply vector-map vector (vector->list m2))))
   m1))

;; Matrix inversion code

(define (get mat i j)
  (vector-ref (vector-ref mat i) j))

(define (set mat i j value)
  (vector-set! (vector-ref mat i) j value))

(define (swap-rows mat i j)
  (let ((temp-row (vector-ref mat i)))
    (vector-set! mat i (vector-ref mat j))
    (vector-set! mat j temp-row)))

(define (scale-row mat i factor)
  (let ((n (vector-length (vector-ref mat 0))))
    (do ((j 0 (+ j 1)))
        ((= j n))
      (set mat i j (* factor (get mat i j))))))

(define (add-scaled-row mat dest src factor)
  (let ((n (vector-length (vector-ref mat 0))))
    (do ((j 0 (+ j 1)))
        ((= j n))
      (set mat dest j (+ (get mat dest j) (* factor (get mat src j)))))))

(define (invert-matrix mat)
  (let* ((n (vector-length mat))
         (identity (make-matrix n n)))
    (do ((i 0 (+ i 1)))
        ((= i n))
      (vector-set! identity i (make-vector n 0))
      (set identity i i 1))
    
    (do ((i 0 (+ i 1)))
        ((= i n))
      (let ((pivot (get mat i i)))
        (if (= pivot 0)
            (let loop ((j (+ i 1)))
              (if (< j n)
                  (if (not (= (get mat j i) 0))
                      (begin (swap-rows mat i j)
                             (swap-rows identity i j))
                      (loop (+ j 1)))))
            (begin (scale-row mat i (/ 1 pivot))
                   (scale-row identity i (/ 1 pivot))))
        
        (do ((j 0 (+ j 1)))
            ((= j n))
          (if (not (= j i))
              (let ((factor (- (get mat j i))))
                (add-scaled-row mat j i factor)
                (add-scaled-row identity j i factor))))))

    identity))

(define (make-matrix rows cols)
  (let ((m (make-vector rows)))
    (do ((i 0 (+ i 1)))
        ((= i rows))
      (vector-set! m i (make-vector cols)))
    m))
