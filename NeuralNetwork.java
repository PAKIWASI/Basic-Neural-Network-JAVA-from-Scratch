public class NeuralNetwork
{
    private final int inputSize;
    private final double[] input;
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

    @Override
    public String toString()
    {
        for ( Layer i : layers )
            i.toString();

        return "";
    }

}
