

#|

Backprop test:

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


(pp a)
(pp b)

|#


#|

(define a (fc-layer 2 2 "RELU"))
(define b (fc-layer 2 1 "RELU"))

(set-layer-weights! a (list (list (default-neuron 1.5) (default-neuron 3.0)) (list (default-neuron 2.2) (default-neuron 4.1))))
(set-layer-biases! a (list (default-neuron 0) (default-neuron 0)))
(set-layer-weights! b (list (list (default-neuron 3.1)) (list (default-neuron 1.6))))
(set-layer-biases! b (list (default-neuron 0)))

(define net (build-network (list a b)))

(define out (car ((run-network net) (list (default-neuron 2.0) (default-neuron 7.0)))))

(n:backward out)

((gradient-step net) 0.01))

|#


(define (setup-network)
  (define a (fc-layer 2 2 "RELU"))
  (define b (fc-layer 2 1 "RELU"))

  (set-layer-weights! a (list (list (default-neuron 1.5) (default-neuron 3.0)) (list (default-neuron 2.2) (default-neuron 4.1))))
  (set-layer-biases! a (list (default-neuron 0) (default-neuron 0)))
  (set-layer-weights! b (list (list (default-neuron 3.1)) (list (default-neuron 1.6))))
  (set-layer-biases! b (list (default-neuron 0)))
  
  (define net (build-network (list a b)))
  net)

(define (setup-network2)
  (define a (fc-layer 2 2 "RELU"))
  (define b (fc-layer 2 1 "RELU"))
  (define net (build-network (list a b)))
net)

(define (bin-network)
  (define a (fc-layer 2 2 "RELU"))
  (define b (fc-layer 2 1 "SIGMOID"))
  (define net (build-network (list a b)))
  net)

#|
Example: setup with preset weights

(define net (setup-network))
((train-network net) train-input train-output 4 0.01 mse-loss)
|#

#|
Example: setup with random weights

(define net (setup-network2))
((train-network net) train-input train-output 1 0.01 mse-loss)
|#

#|
Example: XOR

(define bin-net (bin-network))
((train-network bin-net) bin-input bin-output 1 0.02 mse-loss)

(pp (car ((run-network bin-net) (list (default-neuron 1.5) (default-neuron 1.5)))))
|#
