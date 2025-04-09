public class MatrixOperations
{
    public static double[] MatrixVecXply( double[][] matrix, double[] vector , double[] output)
    {
        int m = matrix.length;
        int n = matrix[ 0 ].length;
        int p = vector.length;

        if ( n != p )
        {
            System.err.println( "Illegel Matrix Multiplication: In m x n xply v x 1, n != v" );
            return null;
        }

        

        for ( int i = 0; i < m; i++ )
        
            for ( int j = 0; j < n; j++ )
            
                output[ i ] += matrix[ i ][ j ] * vector[ j ];
            
        
        return output;
    }    


    // Multiply matrix (transposed) by vector: Wᵀ · gradient
    public static double[] matrixVectorMultiplyTranspose(double[][] matrix, double[] vector)
    {
        double[] result = new double[matrix[0].length];

        for (int i = 0; i < matrix[0].length; i++)

            for (int j = 0; j < vector.length; j++)
             
                result[i] += matrix[j][i] * vector[j];
            
        
        return result;
    }
    
}
