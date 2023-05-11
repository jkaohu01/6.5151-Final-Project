

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

((run-network bin-net) (list (default-neuron 1.5) (default-neuron 1.5)))
|#

#|
(define mnist-net (mnist-network))
((train-network mnist-net) mnist-input mnist-output 1 0.02 mse-loss)

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
  
(define (mnist-network)
  (define a (fc-layer 100 5 "RELU"))
  (define b (fc-layer 5 3 "RELU"))
  (define c (fc-layer 3 1 "SIGMOID"))
  (define net (build-network (list a b c)))
  net)

(define (bin-network)
  (define a (fc-layer 2 2 "RELU"))
  (define b (fc-layer 2 1 "SIGMOID"))
  (define net (build-network (list a b)))
  net)

(define train-input
  (list (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))))

(define train-output
  (list (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))))

