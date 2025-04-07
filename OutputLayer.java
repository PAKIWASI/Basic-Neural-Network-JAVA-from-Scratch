public class OutputLayer extends Layer
{

    public OutputLayer( double[] input, int layersize )
    {
        super( input, layersize );
        
    }

    @Override
    protected double[] activation( double[] intermediate ) // softmax activation
    {
        

        return intermediate;        // TO DO
    }

    @Override
    protected double[] activationDerivative( double[] z )
    {

        return z;         // TO DO
    }
    
}
