# ML Lib

Library to perform ML computations

Example with 2 inputs, 1 hidden layer of 3 neurons and 2 outputs:
```go
net := NewNetwork(2, 3, 2)
outputs := make([]float32, 2)
net.Compute(outputs, inputs)
```
