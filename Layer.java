import java.util.Random;

public abstract class Layer 
{

    protected final double LEARNING_RATE = 0.01; 

    private static Random rand = new Random();
    
    private double[] input;
    private double[][] weights;
    private double[] bias;
    private double[] preActivationOutput; // for backprop
    private double[] output; 

    private double[] localGradient;

    private final int inputSize;
    private final int layerSize;


    public Layer( double[] input, int inputSize, int layerSize )
    {
        this.inputSize = inputSize;
        this.layerSize = layerSize;

        this.input = input;              // reference to input vector, memory for it allocated in respective layer
        localGradient = new double[ layerSize ];
        preActivationOutput = new double[ layerSize ];

        bias = new double[ layerSize ];
        weights = new double[ layerSize ][ inputSize ];
        output = new double[ layerSize ];
        

        initBias( rand );
        initWeights( rand );
    }

    private void initBias( Random rand )
    {
        for ( int i = 0; i < layerSize; i++ )
        
            bias[ i ] = rand.nextGaussian();
    }

    private void initWeights( Random rand )
    {
        for ( int i = 0; i < layerSize; i++ )
        
            for ( int j = 0; j < inputSize; j++ )
            
                weights[ i ][ j ]  = rand.nextGaussian();
    }


    protected abstract double[] activation( double[] intermediate );

    protected abstract double[] activationDerivative( double[] z );


    public void calculateOutput() // z = Wx + b, a = sigma(z)
    {
        double[] intermediate = MatrixOperations.MatrixVecXply( weights, input , output );

        for ( int i = 0; i < layerSize; i++ )
        
            intermediate[ i ] += bias[ i ];


        preActivationOutput = intermediate;
        output = activation( intermediate ); 
    }

    // Computes ∂L/∂W, ∂L/∂b, and ∂L/∂input (for previous layer)
    public abstract void updateParameters( double[] upstreamGradient );

    // Helper: Compute local gradient (∂L/∂z)
    protected double[] computeLocalGradient( double[] upstreamGradient )
    {
        double[] derivative = activationDerivative( preActivationOutput );

        double[] localGradient = new double[ upstreamGradient.length ];

        for ( int i = 0; i < upstreamGradient.length; i++ )
            
            localGradient[ i ] = upstreamGradient[ i ] * derivative[ i ];
        

        return localGradient;
    }


    public double[] getOutput() {  return output;  }
    
    protected double[] getInput() { return input; }
    
    protected double[] getBias() { return bias; }

    public double[][] getweights() { return weights; }

    public int getOutputSize() {  return layerSize;  }

    public int getInputSize() { return inputSize; }

    public void setInput(double[] input) { this.input = input; }

    
    
    @Override
    public String toString()
    {
        System.out.print( "Input Vector : [ " );
        for ( int i = 0; i < inputSize; i++ )
            System.out.print( input[ i ] + " " );
        System.out.print( "]\n" );

        System.out.print( " Weight Matrix :\n" );
        for ( int i = 0; i < layerSize; i++ )
        {
            System.out.print( "[ " );
            for ( int j = 0; j < inputSize; j++ )
                System.out.print( weights[ i ][ j ] + " " );
            System.out.print( "]\n" );
        }

        System.out.print( " Bias Vector : [ " );
        for ( int i = 0; i < layerSize; i++ )
            System.out.print( bias[ i ] + " " );
        System.out.print( "]\n" );

        System.out.print( " Output Vector : [ " );
        for ( int i = 0; i < layerSize; i++ )
            System.out.print( output[ i ] + " " );
        System.out.print( "]\n" );
        
        System.out.println( "----------------------------------------------------------------\n" );

        return "";
    }
    
}
