public class MatrixOperations
{
    public static double[] MatrixVecXply( double[][] matrix, double[] vector )
    {
        int m = matrix.length;
        int n = matrix[ 0 ].length;
        int p = vector.length;

        if ( n != p )
        {
            System.err.println( "Illegel Matrix Multiplication: In m x n xply v x 1, n != v" );
            return null;
        }
                               // actual memory for the output array of each layer
        double[] output = new double[ m ];  // m x n * n x p = m x p = m x 1
        

        for ( int i = 0; i < m; i++ )
        
            for ( int j = 0; j < n; j++ )
            
                output[ i ] += matrix[ i ][ j ] * vector[ j ];
            
        
        return output;
    }    
}
