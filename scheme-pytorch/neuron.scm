


(define-record-type <neuron>
  (make-neuron value grad backward children operation)
  neuron?
  (value neuron-value set-neuron-value!)
  (grad neuron-grad set-neuron-grad!)
  (backward neuron-backward set-neuron-backward!)
  (children neuron-prev set-neuron-prev!)
  (operation neuron-op set-neuron-op!))


(define (default-neuron value)
  (make-neuron value 0.0 (lambda () 'None) (list) (list)))

(define (operation-neuron value children operation)
  (make-neuron value 0.0 (lambda () 'None) children operation))



(define (n:+ neuron1 neuron2)
  (assert (neuron? neuron1))
  (let ((neuron2 (if (neuron? neuron2)
                     neuron2
                     (default-neuron neuron2))))
    (let ((output (operation-neuron (+ (neuron-value neuron1) (neuron-value neuron2)) (list neuron1 neuron2) '+)))
      (let ((backward (lambda ()
                        (set-neuron-grad! neuron1 (+ (neuron-grad neuron1) (neuron-grad output)))
                        (set-neuron-grad! neuron2 (+ (neuron-grad neuron2) (neuron-grad output))))))
        (set-neuron-backward! output backward)
        output))))


(define (n:* neuron1 neuron2)
  (assert (neuron? neuron1))
  (let ((neuron2 (if (neuron? neuron2)
                     neuron2
                     (default-neuron neuron2))))
    (let ((output (operation-neuron (* (neuron-value neuron1) (neuron-value neuron2)) (list neuron1 neuron2) '*)))
      (let ((backward (lambda ()
                        (set-neuron-grad! neuron1 (+ (neuron-grad neuron1) (* (neuron-value neuron2) (neuron-grad output))))
                        (set-neuron-grad! neuron2 (+ (neuron-grad neuron2) (* (neuron-value neuron1) (neuron-grad output)))))))
        (set-neuron-backward! output backward)
        output))))


(define (n:relu neuron1)
  (assert (neuron? neuron1))
  (let ((output (if (>= (neuron-value neuron1) 0)
                    (operation-neuron (neuron-value neuron1) (list neuron1) 'RELU)
                    (operation-neuron 0.0 (list neuron1) 'RELU))))
    (let ((backward (lambda ()
                      (set-neuron-grad! neuron1 (+ (neuron-grad neuron1) (* (if (> (neuron-value output) 0)
                                                                                1.0
                                                                                0.0)
                                                                            (neuron-grad output)))))))
      (set-neuron-backward! output backward)
      output)))

(define (sigmoid x)
  (/ 1.0 (+ 1.0 (exp (- x)))))

(define (sigmoid-derivative x)
  (let ((sig (sigmoid x)))
    (* sig (- 1 sig))))

(define (n:sigmoid neuron1)
  (assert (neuron? neuron1))
  (let ((output (operation-neuron (sigmoid (neuron-value neuron1)) (list neuron1) 'SIGMOID)))
    (let ((backward (lambda ()
                      (set-neuron-grad! neuron1 (+ (neuron-grad neuron1) (* (sigmoid-derivative (neuron-value neuron1))
                                                                              (neuron-grad output)))))))
      (set-neuron-backward! output backward)
      output)))

(define (n:-log neuron1)
  (assert (neuron? neuron1))
  (let ((output (operation-neuron (- (log (neuron-value neuron1))) (list neuron1) '-LOG)))
    (let ((backward (lambda ()
                      (set-neuron-grad! neuron1 (+ (neuron-grad neuron1) (/ (- (neuron-grad output)) (neuron-value neuron1)))))))
      (set-neuron-backward! output backward)
      output)))


(define (reverse lst)
  (if (null? lst)
      '()
      (append (reverse (cdr lst)) (list (car lst)))))

(define (execute-procedures-in-order procedures)
  (map (lambda (procedure) (apply procedure '())) procedures))

(define (n:backward neuron)
  (let ((topo-order '())
        (visited '()))
    (define (build_topo v)
      (if (not (member v visited))
          (let ()
            (set! visited (append visited (list v)))
            (map build_topo (neuron-prev v))
            (set! topo-order (append topo-order (list v))))))
    (build_topo neuron)

    (set-neuron-grad! neuron 1.0)
    (execute-procedures-in-order (map neuron-backward (reverse topo-order)))))
    
          

#|
(load "load-spec")
(define b (default-neuron 1))
(define c (default-neuron 2))
(define a (n:+ b c))

(set-neuron-grad! a 1.0)
(neuron-backward a)
|#

#|
Topo sort test:

(load "load-spec")
(define a (default-neuron 5))
(define b (default-neuron 6))
(define c (n:* a b))

(n:backward c)

3 error> (pp a)
#[neuron 148]
(value 5)
(grad 6.)
(backward #[compound-procedure 151])
(children ())
(operation ())
;Unspecified return value

3 error> (pp b)
#[neuron 149]
(value 6)
(grad 5.)
(backward #[compound-procedure 152])
(children ())
(operation ())
;Unspecified return value

3 error> (pp c)
#[neuron 150]
(value 30)
(grad 1.)
(backward #[compound-procedure 153])
(children (#[neuron 148] #[neuron 149]))
(operation *)
;Unspecified return value               ;


Topo sort test 2:

(load "load-spec")
(define a (default-neuron 5))
(define b (default-neuron 6))
(define c (n:* a b))
(define d (default-neuron 8))
(define e (n:* d c))

(n:backward e)

3 error> (pp a)
#[neuron 158]
(value 5)
(grad 48.)
(backward #[compound-procedure 159])
(children ())
(operation ())
;Unspecified return value

3 error> (pp b)
#[neuron 160]
(value 6)
(grad 40.)
(backward #[compound-procedure 161])
(children ())
(operation ())
;Unspecified return value

3 error> (pp c)
#[neuron 156]
(value 30)
(grad 8.)
(backward #[compound-procedure 162])
(children (#[neuron 158] #[neuron 160]))
(operation *)
;Unspecified return value

3 error> (pp d)
#[neuron 157]
(value 8)
(grad 30.)
(backward #[compound-procedure 163])
(children ())
(operation ())
;Unspecified return value

3 error> (pp e)
#[neuron 154]
(value 240)
(grad 1.)
(backward #[compound-procedure 155])
(children (#[neuron 157] #[neuron 156]))
(operation *)
;Unspecified return value

|#


#|

Backprop test

(define a (default-neuron -4.0))
(define b (default-neuron 2.0))
(define c (n:+ a b))
(define d (n:+ (n:* a b) (n:* (n:* b b) b)))
(define c (n:+ c (n:+ c 1)))
(define c (n:+ c (n:+ (n:+ c 1) (n:* a -1))))
(define d (n:+ d (n:+ (n:* d 2) (n:relu (n:+ b a)))))
(define d (n:+ d (n:+ (n:* d 3) (n:relu (n:+ b (n:* a -1))))))
(define e (n:+ c (n:* d -1)))
(define f (n:* e e))
(define g (n:* f 0.5))
(n:backward g)

1 ]=> (pp a)
#[neuron 619]
(value -4.)
(grad 140.)
(backward #[compound-procedure 620])
(children ())
(operation ())
;Unspecified return value

1 ]=> (pp b)
#[neuron 621]
(value 2.)
(grad 651.)
(backward #[compound-procedure 622])
(children ())
(operation ())
;Unspecified return value

|#
