
public class Main           // SOFTMAX IN OUTPUT LAYER
{
    public static void main( String[] args )
    {
        double[][] matrix = { { 1, 2, 3 },
                              { 4, 5, 6 },
                              { 7, 8, 9 } };
    
        double[] vector = { 2,
                            3,
                            1 };

        int[] hidden = { 2 , 3};
        
        NeuralNetwork nn = new NeuralNetwork( vector, hidden, 5);

        System.out.println(nn);

        nn.forwardPass();

        System.out.println(nn);

    }
}