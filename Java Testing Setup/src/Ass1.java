
import java.io.File;  
import java.io.FileNotFoundException;  
import java.util.Scanner;
import java.util.Vector;

public class Ass1 
{
	static int CountRowMismatches(Vector<String> first, Vector<String> second) 
	{
		int numberOfMismatches = 0;

		for (int i = 0; i < first.size(); i++)
			if (first.get(i).charAt(2) != second.get(i).charAt(0))
				numberOfMismatches++;

		return numberOfMismatches;
	}
	
	static int CountColumnMismatches(Vector<String> first, Vector<String> second) 
	{
		int numberOfMismatches = 0;

		for (int i = 0; i < first.size(); i++)
			if (first.get(i).charAt(1) != second.get(i).charAt(3))
				numberOfMismatches++;

		return numberOfMismatches;
	}
	
	public static void main(String[] args) 
	{
		Vector<Vector<String>> puzzlePieces = new Vector<Vector<String>>();
		
		try 
		{
			String inputFilePath = "..\\Ass1Output.txt";
			
			File inputFile = new File(inputFilePath);
			Scanner myLineReader = new Scanner(inputFile);
		     
			if (myLineReader.hasNextLine())
				myLineReader.nextLine();
		      
		    while (myLineReader.hasNextLine()) 
		    {
		    	Vector<String> row = new Vector<String>();
		        
		    	Scanner myWordReader = new Scanner(myLineReader.nextLine());
		    	 
		    	while (myWordReader.hasNext()) 
		    		row.add(myWordReader.next());
		        
		    	puzzlePieces.add(row);
		    }
		      
		    myLineReader.close();
		} 
		catch (FileNotFoundException e) 
		{
			System.out.println("An error occurred.");
		    e.printStackTrace();
		}
		 
		final int RowSize = 8;
	    final int ColumnSize = 8;
		 
		int numberOfMismatches = 0;
		 
		for (int i = 0; i < RowSize - 1; i++)
			numberOfMismatches += CountRowMismatches(puzzlePieces.get(i), puzzlePieces.get(i + 1));
			
		for (int i = 0; i < ColumnSize - 1; i++)
	    {
			Vector<String> firstColumn = new Vector<String>();
			Vector<String> secondColumn = new Vector<String>();

			for (int j = 0; j < RowSize; j++)
			{
				firstColumn.add(puzzlePieces.get(j).get(i));
				secondColumn.add(puzzlePieces.get(j).get(i + 1));
			}
            
			numberOfMismatches += CountColumnMismatches(firstColumn, secondColumn);
	    }
    	    	 
		System.out.println("Number of mismatches : " + numberOfMismatches);
	}	

}
