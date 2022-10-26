import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

public class data {
public static void main(String[] args) throws IOException{

    FileWriter outStream = new FileWriter("data.csv");
    PrintWriter printer = new PrintWriter(outStream);

    FileInputStream inStream = new FileInputStream("/Users/shanestevenson/Downloads/creditcard.csv");
    Scanner scnr = new Scanner(inStream);
    scnr.nextLine();


    int count = 0;
    int zeroes = 0;
    int ones = 0;
    while(scnr.hasNextLine()) {
        String str = scnr.nextLine();
        String[] row = str.split(",");
        if(row[row.length-1].equals("\"0\"") && count >= 578) {
            zeroes++;
            printer.println(str);
            count = 0;
        } else if(row[row.length-1].equals("\"1\"")) {
            ones++;
            printer.println(str);
        }
        count++;

    }
    System.out.println("zeroes: " + zeroes);
    System.out.println("ones : " + ones);

    inStream.close();
    scnr.close();
    outStream.close();

    }
}