using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;

// Using CSV
using CsvHelper;
using System.Collections;

// Using InputBox
using Microsoft.VisualBasic;
using System.Drawing.Imaging;

using System.Windows.Media.Imaging;

namespace Image_Classification_Tool
{
    public partial class Form1 : Form
    {
        public static int fileCount;
        public static int index;
        public static string dir;
        public static string csvPath;

        public static int btnCount = 0;

        //csv class
        public class CsvWrite
        {
            public string FileName { get; set; }
            public string Result { get; set; }
        }

        public Form1()
        {
            InitializeComponent();
            this.StartPosition = FormStartPosition.CenterScreen;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            goodbtn.Enabled = false;
            badbtn.Enabled = false;
            //button1.Visible = false;
            //pictureBox1.Left = (this.ClientSize.Width - pictureBox1.Width) / 2;
            //pictureBox1.Top = (this.ClientSize.Height - pictureBox1.Height) / 2;
        }

        //建立空白csv
        private void CreatCsv()
        {
            string user = Interaction.InputBox("Enter your name.", "Please enter user name", "輸入名稱", -1, -1);
            csvPath = dir + user + ".csv";
            Console.WriteLine(csvPath);
            // 建立空的CSV檔案
            File.WriteAllText(csvPath, string.Empty);
            // 欄位格式為 FileName, user
            using (var writer = new StreamWriter(csvPath, true, Encoding.UTF8))
            { writer.WriteLine($"FileName,{user}"); }
        }

        // 切換圖片
        private void ChangeImage()
        {
            DirectoryInfo di = new DirectoryInfo(dir);
            FileInfo[] files = di.GetFiles("*.tif"); //篩選副檔名
            int fileCount = files.Length; //取得個數個數
            //Console.WriteLine(fileCount); //顯示檔案個數

            string[] files2 = Directory.GetFiles(dir, "*.tif"); // 讀取路徑下所有檔案名稱
            foreach (var file in files2)
            {
                //Console.WriteLine(file);
            }

            string[] FilesInfo = new string[files.Length]; // 設置array大小為檔案個數
            string[] tmp = new string[files.Length];

            for (int i = 0; i < files.Length; i++)
            {
                FilesInfo[i] = dir + files[i].ToString(); // 設置圖片檔路徑
                //Console.WriteLine(FilesInfo[i]);
                tmp[i] = dir + files[i].ToString();
            }

            if (index == files.Length - 1) // 在顯示最後一張時將button停止防止error
            {
                goodbtn.Enabled = false;
                badbtn.Enabled = false;
            }
            else
            {
                index++;
            }


            pictureBox1.Image = Image.FromFile(FilesInfo[index]);
            pictureBox2.Image = Image.FromFile(tmp[index]);

        }

        // as to which folder then change image + write into csv
        private void ToFolderCsv(string str)
        {
            string sourceFilePath = dir; // 檔案路徑
            string destinationFolderPath = dir + str; // 目標資料夾路徑

            // 確認目標資料夾存在，如果不存在則建立新資料夾
            if (!Directory.Exists(destinationFolderPath))
            {
                Directory.CreateDirectory(destinationFolderPath);
            }

            // 取得檔案名稱並構建新路徑
            //string fileName = Path.GetFileName(sourceFilePath);         
            //string destinationFilePath = Path.Combine(destinationFolderPath, fileName);

            string[] strFiles = new string[fileCount];
            DirectoryInfo di = new DirectoryInfo(dir);
            FileInfo[] files = di.GetFiles("*.tif"); //篩選副檔名       

            string[] FilesInfo = new string[fileCount]; // 設置array大小為檔案個數

            for (int i = 0; i < fileCount; i++)
            {
                FilesInfo[i] = dir + files[i].ToString(); // 設置圖片檔路徑
                //Console.WriteLine(FilesInfo[i]);
            }

            for (int i = 0; i < fileCount; i++)
            {
                strFiles[i] = destinationFolderPath + files[i].ToString(); // 設置圖片檔新路徑
                //Console.WriteLine(FilesGood[i]);
            }

            // 將檔案複製到新資料夾中
            File.Copy(FilesInfo[index], strFiles[index]);
            //System.IO.File.Copy(sourceFilePath, destinationFilePath);   
            ChangeImage(); //變更顯示圖片

            // 將檔名和result寫入csv
            List<CsvWrite> csvWrites = new List<CsvWrite>();
            if (btnCount < fileCount)
            {
                csvWrites.Add(new CsvWrite { FileName = files[btnCount].ToString(), Result = "good" });
            }

            using (var writer = new StreamWriter(csvPath, true, Encoding.UTF8))
            {
                foreach (var csvWrite in csvWrites)
                {
                    string row = $"{csvWrite.FileName},{csvWrite.Result}";
                    writer.WriteLine(row);
                }
            }

            btnCount++;
        }

        // User Manual
        private void button2_Click_1(object sender, EventArgs e)
        {
            string message = "Manual:\n說明此影像分類工具之功能\n\nUpload Photo按鈕 - 按下後將跳出選擇資料夾畫面, 可自行匯入資料夾, 匯入後將會顯示第一張影像\nTips: 資料夾內限定影像檔要是 *.tif檔, 使用者輸入名稱後會自動產生csv檔\n\n好 按鈕 -若判斷為當前圖片為好, 按下後將會自動將影像分類至 good資料夾\nTips: 若無資料夾則會自動生成\n\n壞 按鈕 - 若判斷為當前圖片為壞, 按下後將會自動將影像分類至 bad資料夾\nTips: 若無資料夾則會自動生成\n\n此程式會自動生成一個csv檔, 分類時會將檔案名稱及判斷結果寫入檔案";
            MessageBox.Show(message);
        }

        // Upload Button
        private void button2_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog path = new FolderBrowserDialog();
            path.ShowDialog(); // 顯示資料夾來提供使用者選擇
            //Console.WriteLine(path.SelectedPath); // 存取選擇資料夾路徑

            Console.WriteLine("TEST 有無抓到檔案");

            dir = path.SelectedPath + "\\";

            DirectoryInfo di = new DirectoryInfo(dir); // 抓資料夾path
            FileInfo[] files = di.GetFiles("*.tif"); //篩選副檔名
            fileCount = files.Length; //取得個數個數
            Console.WriteLine(fileCount); //顯示檔案個數

            string[] files2 = Directory.GetFiles(dir, "*.tif"); // 讀取路徑下所有檔案名稱
            foreach (var file in files2)
            {
                Console.WriteLine(file);
            }

            string[] FilesInfo = new string[files.Length]; // 設置array大小為檔案個數
            string[] tmp = new string[files.Length];

            for (int i = 0; i < files.Length; i++)
            {
                FilesInfo[i] = dir + files[i].ToString(); // 設置圖片檔路徑
                //Console.WriteLine(FilesInfo[i]);
                tmp[i] = dir + files[i].ToString(); // 暫存圖片檔路徑
            }

            CreatCsv();

            index = 0; // 預設起始值為0 (第一張影像)
            pictureBox1.Image = Image.FromFile(FilesInfo[index]);
            pictureBox2.Image = Image.FromFile(tmp[index]);

            goodbtn.Enabled = true;
            badbtn.Enabled = true;

        }

        // to good folder
        private void button1_Click(object sender, EventArgs e)
        {
            string path = "good\\";
            ToFolderCsv(path);

        }

        // to bad folder
        private void button3_Click(object sender, EventArgs e)
        {
            string path = "bad\\";
            ToFolderCsv(path);
        }

        // Sharpen
        private void button3_Click_1(object sender, EventArgs e)
        {
            string sourceFilePath = dir; // 檔案路徑
            string destinationFolderPath = dir + "Sharpen filter\\"; // 目標資料夾路徑

            // 確認目標資料夾存在，如果不存在則建立新資料夾
            if (!Directory.Exists(destinationFolderPath))
            {
                Directory.CreateDirectory(destinationFolderPath);
            }

            // 取得檔案名稱並構建新路徑
            //string fileName = Path.GetFileName(sourceFilePath);         
            //string destinationFilePath = Path.Combine(destinationFolderPath, fileName);

            string[] strFiles = new string[fileCount];
            DirectoryInfo di = new DirectoryInfo(dir);
            FileInfo[] files = di.GetFiles("*.tif"); //篩選副檔名       

            string[] FilesInfo = new string[fileCount]; // 設置array大小為檔案個數

            for (int i = 0; i < fileCount; i++)
            {
                FilesInfo[i] = dir + files[i].ToString(); // 設置圖片檔路徑
                //Console.WriteLine(FilesInfo[i]);
            }

            for (int i = 0; i < fileCount; i++)
            {
                strFiles[i] = destinationFolderPath + files[i].ToString(); // 設置圖片檔新路徑
                //Console.WriteLine(FilesGood[i]);
            }

            pictureBox2.Image = Image.FromFile(FilesInfo[index]); // init

            // Sharpen
            try
            {
                int Height = this.pictureBox1.Image.Height;
                int Width = this.pictureBox1.Image.Width;
                Bitmap newBitmap = new Bitmap(Width, Height);
                Bitmap oldBitmap = (Bitmap)this.pictureBox1.Image;
                Color pixel;
                //拉普拉斯模板
                int[] Laplacian = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
                for (int x = 1; x < Width - 1; x++)
                    for (int y = 1; y < Height - 1; y++)
                    {
                        int r = 0, g = 0, b = 0;
                        int Index = 0;
                        for (int col = -1; col <= 1; col++)
                            for (int row = -1; row <= 1; row++)
                            {
                                pixel = oldBitmap.GetPixel(x + row, y + col); r += pixel.R * Laplacian[Index];
                                g += pixel.G * Laplacian[Index];
                                b += pixel.B * Laplacian[Index];
                                Index++;
                            }
                        //處理顏色值溢出
                        r = r > 255 ? 255 : r;
                        r = r < 0 ? 0 : r;
                        g = g > 255 ? 255 : g;
                        g = g < 0 ? 0 : g;
                        b = b > 255 ? 255 : b;
                        b = b < 0 ? 0 : b;
                        newBitmap.SetPixel(x - 1, y - 1, Color.FromArgb(r, g, b));
                    }
                this.pictureBox2.Image = newBitmap;

                Image imageToSave = pictureBox2.Image;
                string imagepath = destinationFolderPath + "Sharpen" + btnCount.ToString() + ".tif";
                Console.WriteLine(imagepath);

                // Create an EncoderParameters object to specify the TIFF compression method
                EncoderParameters encoderParams = new EncoderParameters(1);
                encoderParams.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Compression, (long)EncoderValue.CompressionLZW);

                using (var stream = new FileStream(imagepath, FileMode.Create))
                {
                    TiffBitmapEncoder encoder = new TiffBitmapEncoder();
                    encoder.Compression = TiffCompressOption.Zip;

                    // Convert Image to BitmapImage
                    MemoryStream ms = new MemoryStream();
                    imageToSave.Save(ms, ImageFormat.Png);
                    BitmapImage bitmapImage = new BitmapImage();
                    bitmapImage.BeginInit();
                    bitmapImage.StreamSource = new MemoryStream(ms.ToArray());
                    bitmapImage.EndInit();

                    // Convert BitmapImage to BitmapFrame
                    BitmapFrame frame = BitmapFrame.Create(bitmapImage);

                    encoder.Frames.Add(frame);
                    encoder.Save(stream);
                }
            }
            catch (Exception ex)
            {
                // 處理未處理的例外狀況
                Console.WriteLine("Exception occurred: {0}", ex.Message);
            }

        }


        private void button1_Click_1(object sender, EventArgs e)
        {
            // for testing
            pictureBox2.Image = Image.FromFile("C:\\Users\\USER\\Downloads\\123.jpg");
        }

        // Histogram Equlization
        private void button2_Click_2(object sender, EventArgs e)
        {
            string sourceFilePath = dir; // 檔案路徑
            string destinationFolderPath = dir + "HistoEq filter\\"; // 目標資料夾路徑

            // 確認目標資料夾存在，如果不存在則建立新資料夾
            if (!Directory.Exists(destinationFolderPath))
            {
                Directory.CreateDirectory(destinationFolderPath);
            }

            // 取得檔案名稱並構建新路徑
            //string fileName = Path.GetFileName(sourceFilePath);         
            //string destinationFilePath = Path.Combine(destinationFolderPath, fileName);

            string[] strFiles = new string[fileCount];
            DirectoryInfo di = new DirectoryInfo(dir);
            FileInfo[] files = di.GetFiles("*.tif"); //篩選副檔名       

            string[] FilesInfo = new string[fileCount]; // 設置array大小為檔案個數

            for (int i = 0; i < fileCount; i++)
            {
                FilesInfo[i] = dir + files[i].ToString(); // 設置圖片檔路徑
                //Console.WriteLine(FilesInfo[i]);
            }

            for (int i = 0; i < fileCount; i++)
            {
                strFiles[i] = destinationFolderPath + files[i].ToString(); // 設置圖片檔新路徑
                //Console.WriteLine(FilesGood[i]);
            }

            pictureBox2.Image = Image.FromFile(FilesInfo[index]); // init

            
        }

    }
}
