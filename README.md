# Learning Go the Hard Way

> Nothing in life should come easy. I taught myself Go(lang) by rebuilding the Multi-layer Perceptron (MLP) neural net I coded a few months ago [in Python](https://github.com/anthonykrivonos/mlp).

My notes and findings from my journey will be written in this README. Everything else can be found in the code.

## Installing and Running


## Notes

### 1. Creating a New Project

We'll start by creating a new Go module in the `/mlp` directory.

```
mkdir mlp && go mod init mlp`
```

Next, we'll create a new file for the neural network and add some boilerplate to it.

By convention, files are underscore-cased.
```
touch mlp/neural_net.go
```

Let's add some boilerplate to this file:
```
package main

import "fmt"

func main() {
	// our code goes here
	fmt.Println("this is where our code will go")
}
```

Sanity check––compile and run the code via the following line.
```
go run mlp/neural_net.go
```
> `this is where our code will go`

After noticing the pun in the output, I decided to create the `Loss`, `LayerType`, `Regularization`, and `Activation` enum types first. These enums will help dictate how each section of our neural network will operate, and will teach us both about enums and importing modules into a main module.

### 2. Creating Enum Submodules



## References

## Author

Anthony Krivonos ([GitHub](https://github.com/anthonykrivonos) | [Portfolio](https://anthonykrivonos.com))

