public class HiddenLayer extends Layer
{

    public HiddenLayer( double[] input, int inputsize, int layersize )
    {
        super( input, inputsize, layersize );
    }


    @Override
    protected double[] activation( double[] intermediate )  // ReLu
    {            
        for ( int i = 0; i < intermediate.length; i++ )
    
            if ( intermediate[ i ] < 0 )
                
                intermediate[ i ] = 0;
        

        return intermediate;
    }

    @Override
    protected double[] activationDerivative( double[] z )
    {
        for ( int i = 0; i < z.length; i++ )
        {
            if ( z[ i ] <= 0 )
                z[ i ] = 0;      
            else    
                z[ i ] = 1;
        }

        return z;
    }

    @Override
    public void updateParameters(double[] upstreamGradient)
    {
        double[] localGradient = computeLocalGradient(upstreamGradient);
        
        double[] input = getInput();
        double[] bias = getBias();
        double[][] weights = getweights();

        // Update weights and biases (gradient descent)
        for (int i = 0; i < getOutputSize(); i++)
        {
            for (int j = 0; j < getInputSize(); j++)
                weights[i][j] -= LEARNING_RATE * localGradient[i] * input[j];
            
            bias[i] -=LEARNING_RATE * localGradient[ i ];
        }
    }
    
}
