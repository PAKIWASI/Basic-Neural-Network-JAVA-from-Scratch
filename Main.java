
public class Main
{
    public static void main( String[] args )
    {
        double[][] matrix = { { 1, 2, 3 },
                              { 4, 5, 6 },
                              { 7, 8, 9 } };
    
        double[] vector = { 2,
                            3,
                            1 };

        int[] hidden = { 2 };
        
        NeuralNetwork nn = new NeuralNetwork( vector, hidden, 3);

        System.out.println(nn);

        nn.forwardPass();

        System.out.println(nn);

    }

}