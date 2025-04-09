public class OutputLayer extends Layer
{
    private double loss;
    private double[] trueOutput;    // 0 for all elements except the true output number
    private double[] finalOutput;


    public OutputLayer( double[] input, int inputSize, int layersize )
    {
        super( input, inputSize, layersize );
        loss = 0;
        
    }

    public void calculateLoss( double[] output, double[] trueOutput )   // cross entropy error
    {
        double sum = 0;
        final double epsilon = 1e-12; // to prevent log(0)
    
        for ( int i = 0; i < output.length; i++ )
            sum += trueOutput[ i ] * Math.log( output[ i ] + epsilon );
        
        loss = -sum;
    }

    @Override
    protected double[] activation( double[] intermediate )  // softmax activation
    {
        double[] expValues = new double[ intermediate.length ];
        double sum = 0;
        
        // Calculate exponentials and sum
        for ( int i = 0; i < intermediate.length; i++ ) 
        {
            expValues[ i ] = Math.exp( intermediate[ i ] );
            sum += expValues[ i ];
        }
        
        // Normalize
        for ( int i = 0; i < intermediate.length; i++ )

            intermediate[ i ] = expValues[ i ] / sum;
        
        return intermediate;
    }

    @Override
    protected double[] activationDerivative( double[] z )
    {
        return calculateOutputGradient( this.getOutput(), trueOutput );
    }

    public double[] calculateOutputGradient( double[] output, double[] trueOutput )
    {
        double[] gradient = new double[ output.length ];
        
        for ( int i = 0; i < output.length; i++ )
            
            gradient[ i ] = output[ i ] - trueOutput[ i ];
        
        return gradient;
    }

    @Override
    public void updateParameters( double[] upstreamGradient )
    {
        // For softmax + cross-entropy, upstreamGradient is already (output - trueOutput)
        double[] localGradient = calculateOutputGradient(getOutput(), trueOutput);
        
        double[] input = getInput();
        double[] bias = getBias();
        double[][] weights = getweights();

        // Update weights and biases
        for (int i = 0; i < getOutputSize(); i++)  // gradient decent
        {
            for (int j = 0; j < getInputSize(); j++)

                weights[i][j] -= LEARNING_RATE * localGradient[i] * input[j];
            
            bias[i] -= LEARNING_RATE * localGradient[i];
        }
    }

    public void setTrueOutput(double[] trueOutput) { this.trueOutput = trueOutput; }
}
