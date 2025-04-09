public class NeuralNetwork
{
    private final int inputSize;
    private double[] input;
    private final Layer[] layers;
    private final double[] finalOutput;


    public NeuralNetwork( double[] input, int[] hidden, int outputSize )
    {
        this.input = input;
        this.inputSize = input.length;
        
        layers = new Layer[ hidden.length + 1 ]; // hidden + output layers, input is just a vector

        layers[ 0 ] = new HiddenLayer( input, inputSize, hidden[ 0 ] );

        for ( int i = 1; i < hidden.length; i++ )
        
            layers[ i ] = new HiddenLayer( layers[ i - 1 ].getOutput(), layers[ i - 1 ].getOutputSize() ,hidden[ i ] );
        

        layers[ hidden.length ] = new OutputLayer( layers[ hidden.length - 1 ].getOutput(), layers[ hidden.length - 1].getOutputSize(), outputSize );

        finalOutput = layers[ hidden.length ].getOutput();
    }

    public void forwardPass()
    {
        for ( Layer l : layers )
            l.calculateOutput(); 
    }

    public void backPropagation(double[] trueOutput)
    {
        // Start with output layer gradient
        double[] gradient = ((OutputLayer) layers[layers.length - 1]).calculateOutputGradient(finalOutput, trueOutput);
        
        // Backpropagate through layers
        for (int i = layers.length - 1; i >= 0; i--)
        {
            layers[i].updateParameters(gradient);
            if (i > 0)
                // Compute gradient for previous layer
                gradient = MatrixOperations.matrixVectorMultiplyTranspose( layers[i].getweights(), gradient );
            
        }
    }

    public void setInput(double[] newInput) {
        this.input = newInput;
        layers[0].setInput(newInput); // Propagate to first hidden layer
    }

    @Override
    public String toString()
    {
        for ( Layer i : layers )
            i.toString();

        return "";
    }
}
