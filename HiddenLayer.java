public class HiddenLayer extends Layer
{

    public HiddenLayer( double[] input, int layersize )
    {
        super( input, layersize );
        
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
        return z; // TO DO
    }
    
}
