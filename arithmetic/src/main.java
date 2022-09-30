import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class main {
    public main() throws IOException {
    }
    public static void main(String[] args) throws IOException {
        ArrayList<String> array1=new ArrayList<String>();
        for (int i = 0; i < args.length; i++) {
            array1.add(args[i]);
        }
        int index_r=array1.indexOf("-r"); // “-r”的索引值   值为-1则不存在
        int index_n=array1.indexOf("-n"); //‘-n’的索引值    值为-1则不存在
        int index_range=0;
        int index_num = 0;
        try {
            index_range=Integer.parseInt(array1.get(index_r+1));
            index_num=Integer.parseInt(array1.get(index_n+1));
        }catch (Exception e){
            System.out.println("触发异常"+e);
            System.out.println("输入参数不对，请重启程序");
        }
        Set<Expression> expressions = new HashSet<Expression>();
        Path exercisesPath = FileSystems.getDefault().getPath("Exercises.txt");
        Path answersPath = FileSystems.getDefault().getPath("Answers.txt");
        BufferedWriter exercisesBufferedWriter = Files.newBufferedWriter(exercisesPath, Charset.forName("UTF-8"));
         BufferedWriter answersBufferedWriter = Files.newBufferedWriter(answersPath, Charset.forName("UTF-8"));
        for(int i = 1;i<=index_num;){
            try {
                Expression expression = new Expression(index_range);
                //如果表达式不重复，则写入输出文件
                if(expressions.add(expression)){
                    exercisesBufferedWriter.write(i +". "+expression.getExpression()+"\n");
                    answersBufferedWriter.write(i +". "+expression.getResult().toString()+"\n");
                    System.out.println(i+"."+expression.getExpression());
                    i++;
                }
            } catch (Exception e) {

            }
        }
        exercisesBufferedWriter.flush();
        exercisesBufferedWriter.close();
        answersBufferedWriter.flush();
        answersBufferedWriter.close();
    }
}
