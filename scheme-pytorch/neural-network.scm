

(define-record-type <layer>
  (make-layer n-in n-out act-func forward weights biases)
  layer?
  (n-in layer-in-size set-in-size!)
  (n-out layer-out-size set-out-size!)
  (act-func layer-act-func set-layer-act-func!)
  (forward layer-forward set-layer-forward!)
  (weights layer-weights set-layer-weights!)
  (biases layer-biases set-layer-biases!))

(define-record-type <network>
  (make-network layers)
  network?
  (layers network-layers set-network-layers!))

(define (random-weight)
  ;; (random 1.0))
  (- (* 2 (random 1.0)) 1))

(define (random-list n)
  (if (zero? n)
      '()
      (cons (random-weight) (random-list (- n 1)))))

(define (random-weights n-in n-out)
  (if (zero? n-in)
      '()
      (cons (map default-neuron (random-list n-out)) (random-weights (- n-in 1) n-out))))

(define (random-biases n)
  (map default-neuron (random-list n)))

(define (multiply-neurons inputs weights)
  (define (dot-product a b)
    (fold-left n:+ (default-neuron 0.0) (map n:* a b)))

  (define (transpose matrix)
    (apply map list matrix))
  
  (map (lambda (row) (dot-product inputs row)) (transpose weights)))

(define (add-biases base biases)
  (assert (= (length base) (length biases)))
  (map n:+ base biases))

(define (chain-functions functions input)
  (define (apply-function result function)
    (function result))
  
  (fold-left apply-function input functions))


(define (forward-pass input layers)
  (let ((layer-forwards (map (lambda (layer)
                               (let ((forward-function (layer-forward layer)))
                                 (forward-function (layer-weights layer) (layer-biases layer) (layer-act-func layer)))) layers)))
    (chain-functions layer-forwards input)))

(define (fc-layer n-in n-out act-func)
  (make-layer n-in n-out act-func (lambda (weights biases act-func)
                                     (lambda (inputs)
                                       (if (equal? act-func "RELU")
                                           (map n:relu (add-biases (multiply-neurons inputs weights) biases))
                                           (if (equal? act-func "SIGMOID")
                                               (map n:sigmoid (add-biases (multiply-neurons inputs weights) biases))
                                               (add-biases (multiply-neurons inputs weights) biases)))))
              (random-weights n-in n-out) (random-biases n-out)))

(define (build-network layers)
  (make-network layers))

(define (run-network network)
  (lambda (input)
    (forward-pass input (network-layers network))))

(define (flatten x)
  (cond ((null? x) '())
        ((pair? x) (append (flatten (car x)) (flatten (cdr x))))
        (else (list x))))

(define (gradient-step network)
  (lambda (step-size)
    (let ((all-weights (flatten (map layer-weights (network-layers network))))
          (all-biases (flatten (map layer-biases (network-layers network)))))
      (map (lambda (neuron)
             (if (and (< (neuron-grad neuron) 5) (> (neuron-grad neuron) -5))
                 (set-neuron-value! neuron (- (neuron-value neuron) (* step-size (neuron-grad neuron))))
                 (if (< (neuron-grad neuron) -5)
                     (set-neuron-value! neuron (- (neuron-value neuron) (* step-size -5)))
                     (set-neuron-value! neuron (- (neuron-value neuron) (* step-size 5)))))
             (set-neuron-grad! neuron 0.0)) all-weights)
      (map (lambda (neuron)
             (if (and (< (neuron-grad neuron) 5) (> (neuron-grad neuron) -5))
                 (set-neuron-value! neuron (- (neuron-value neuron) (* step-size (neuron-grad neuron))))
                 (if (< (neuron-grad neuron) -5)
                     (set-neuron-value! neuron (- (neuron-value neuron) (* step-size -5)))
                     (set-neuron-value! neuron (- (neuron-value neuron) (* step-size 5)))))
             (set-neuron-grad! neuron 0.0)) all-biases))))

#|
(load "load-spec")

(define a (fc-layer 2 2 #f))
(define b (fc-layer 2 1 #f))

(set-layer-weights! a (list (list (default-neuron 1.5) (default-neuron 3.0)) (list (default-neuron 2.2) (default-neuron 4.1))))
(set-layer-biases! a (list (default-neuron 0) (default-neuron 0)))
(set-layer-weights! b (list (list (default-neuron 3.1)) (list (default-neuron 1.6))))
(set-layer-biases! b (list (default-neuron 0)))

(define net (build-network (list a b)))

(define out (car ((run-network net) (list (default-neuron 2.0) (default-neuron 7.0)))))

(n:backward out)

((gradient-step net) 0.01)




(load "load-spec")
(define a (fc-layer 2 2 #f))
(define b (fc-layer 2 1 #f))
(set-layer-weights! a (list (list (default-neuron 1.5) (default-neuron 3.0)) (list (default-neuron 2.2) (default-neuron 4.1))))
(set-layer-biases! a (list (default-neuron 0) (default-neuron 0)))
(set-layer-weights! b (list (list (default-neuron 3.1)) (list (default-neuron 1.6))))
(set-layer-biases! b (list (default-neuron 0)))
(pp a)
(pp b)
(define output (car (forward-pass (list (default-neuron 2) (default-neuron 7)) (list a b))))
(pp output)
(n:backward output)
(pp (layer-weights a))



(load "load-spec")
(define a (fc-layer 3 2 #t))
(define b (fc-layer 2 1 #t))
(forward-pass (list
               (default-neuron 2.0)
               (default-neuron 3.0)
               (default-neuron 4.0))
(list a b))




(multiply-neurons (list (default-neuron 2.0) (default-neuron 3.0)) (list (list (default-neuron 0.2) (default-neuron 0.3)) (list (default-neuron 0.13) (default-neuron 0.67))))


;Value: (#[neuron 38] #[neuron 39])

4 error> (pp #@38)
#[neuron 38]
(value .79)
(grad 0.)
(backward #[compound-procedure 40])
(children (#[neuron 42] #[neuron 41]))
(operation +)
;Unspecified return value

4 error> (pp #@39)
#[neuron 39]
(value 2.6100000000000003)
(grad 0.)
(backward #[compound-procedure 43])
(children (#[neuron 45] #[neuron 44]))
(operation +)
;Unspecified return value               ;


(load "load-spec")
(define a (fc-layer 3 2 #t))
(define b (fc-layer 2 1 #t))
(define output (car (forward-pass (list (default-neuron 2.0) (default-neuron 3.0) (default-neuron 4.0)) (list a b))))
|#

