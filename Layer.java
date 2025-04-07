import java.util.Random;

public abstract class Layer 
{
    private static Random rand = new Random();
    
    private double[] input;
    private double[][] weights;
    private double[] bias;
    private double[] output;

    private final int inputSize;
    private final int layerSize;


    public Layer( double[] input, int layersize )
    {
        this.inputSize = input.length;
        this.layerSize = layersize;

        bias = new double[ layersize ];
        this.input = input;              // reference to input vector, memory for it allocated in respective layer
        weights = new double[ layersize ][ inputSize ];
    
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
                                // intermediae, output memory is allocated in func
        double[] intermediate = MatrixOperations.MatrixVecXply( weights, input );

        for ( int i = 0; i < layerSize; i++ )
        
            intermediate[ i ] += bias[ i ];


        output = activation( intermediate ); 
    }

    public double[] getoutput()
    {
        return output;
    }
    
    @Override
    public String toString()
    {
        System.out.print( "Input Vector : [ " );
        for ( int i = 0; i < inputSize; i++ )
            System.out.print( input[ i ] + " " );
        System.out.print( "]\n\n" );
        
        System.out.print( " Weight Matrix :\n" );
        for ( int i = 0; i < layerSize; i++ )
        {
            System.out.print( "[ " );
            for ( int j = 0; j < inputSize; j++ )
                System.out.print( weights[ i ][ j ] + " " );
            System.out.print( "]\n\n" );
        }

        System.out.println( " Output Vector : [ " );
        for ( int i = 0; i < layerSize; i++ )
            System.out.print( output[ i ] + " " );
        System.out.print( "]\n\n" );
        
        return "";
    }
    
}
