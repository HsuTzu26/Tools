
namespace Image_Classification_Tool
{
    partial class Form1
    {
        /// <summary>
        /// 設計工具所需的變數。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清除任何使用中的資源。
        /// </summary>
        /// <param name="disposing">如果應該處置受控資源則為 true，否則為 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form 設計工具產生的程式碼

        /// <summary>
        /// 此為設計工具支援所需的方法 - 請勿使用程式碼編輯器修改
        /// 這個方法的內容。
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.uploadbtn = new System.Windows.Forms.Button();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.goodbtn = new System.Windows.Forms.Button();
            this.badbtn = new System.Windows.Forms.Button();
            this.manualbtn = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.button1 = new System.Windows.Forms.Button();
            this.pictureBox2 = new System.Windows.Forms.PictureBox();
            this.button2 = new System.Windows.Forms.Button();
            this.button3 = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).BeginInit();
            this.SuspendLayout();
            // 
            // uploadbtn
            // 
            this.uploadbtn.Font = new System.Drawing.Font("Times New Roman", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.uploadbtn.Location = new System.Drawing.Point(1030, 10);
            this.uploadbtn.Name = "uploadbtn";
            this.uploadbtn.Size = new System.Drawing.Size(150, 36);
            this.uploadbtn.TabIndex = 2;
            this.uploadbtn.Text = "Upload Photo\r\n";
            this.uploadbtn.UseVisualStyleBackColor = true;
            this.uploadbtn.Click += new System.EventHandler(this.button2_Click);
            // 
            // pictureBox1
            // 
            this.pictureBox1.Location = new System.Drawing.Point(178, 67);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(400, 400);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox1.TabIndex = 3;
            this.pictureBox1.TabStop = false;
            // 
            // goodbtn
            // 
            this.goodbtn.Font = new System.Drawing.Font("微軟正黑體", 13.8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.goodbtn.Location = new System.Drawing.Point(1080, 111);
            this.goodbtn.Name = "goodbtn";
            this.goodbtn.Size = new System.Drawing.Size(100, 100);
            this.goodbtn.TabIndex = 4;
            this.goodbtn.Text = "Good";
            this.goodbtn.UseVisualStyleBackColor = true;
            this.goodbtn.Click += new System.EventHandler(this.button1_Click);
            // 
            // badbtn
            // 
            this.badbtn.Font = new System.Drawing.Font("微軟正黑體", 13.8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.badbtn.Location = new System.Drawing.Point(1080, 279);
            this.badbtn.Name = "badbtn";
            this.badbtn.Size = new System.Drawing.Size(100, 100);
            this.badbtn.TabIndex = 5;
            this.badbtn.Text = "Bad";
            this.badbtn.UseVisualStyleBackColor = true;
            this.badbtn.Click += new System.EventHandler(this.button3_Click);
            // 
            // manualbtn
            // 
            this.manualbtn.BackColor = System.Drawing.Color.Transparent;
            this.manualbtn.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("manualbtn.BackgroundImage")));
            this.manualbtn.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.manualbtn.FlatAppearance.BorderSize = 0;
            this.manualbtn.FlatAppearance.MouseDownBackColor = System.Drawing.Color.Transparent;
            this.manualbtn.FlatAppearance.MouseOverBackColor = System.Drawing.Color.Transparent;
            this.manualbtn.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.manualbtn.Location = new System.Drawing.Point(20, 10);
            this.manualbtn.Name = "manualbtn";
            this.manualbtn.Size = new System.Drawing.Size(50, 54);
            this.manualbtn.TabIndex = 6;
            this.manualbtn.UseVisualStyleBackColor = false;
            this.manualbtn.Click += new System.EventHandler(this.button2_Click_1);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Times New Roman", 10F, System.Drawing.FontStyle.Bold);
            this.label1.Location = new System.Drawing.Point(12, 67);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(65, 19);
            this.label1.TabIndex = 7;
            this.label1.Text = "Manual";
            // 
            // button1
            // 
            this.button1.Font = new System.Drawing.Font("Times New Roman", 13.8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button1.Location = new System.Drawing.Point(20, 459);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(100, 100);
            this.button1.TabIndex = 8;
            this.button1.Text = "None";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click_1);
            // 
            // pictureBox2
            // 
            this.pictureBox2.Location = new System.Drawing.Point(609, 67);
            this.pictureBox2.Name = "pictureBox2";
            this.pictureBox2.Size = new System.Drawing.Size(400, 400);
            this.pictureBox2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox2.TabIndex = 9;
            this.pictureBox2.TabStop = false;
            // 
            // button2
            // 
            this.button2.Font = new System.Drawing.Font("微軟正黑體", 10F, System.Drawing.FontStyle.Bold);
            this.button2.Location = new System.Drawing.Point(12, 130);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(124, 68);
            this.button2.TabIndex = 10;
            this.button2.Text = "Histogram Eqalization";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click_2);
            // 
            // button3
            // 
            this.button3.Font = new System.Drawing.Font("微軟正黑體", 10F, System.Drawing.FontStyle.Bold);
            this.button3.Location = new System.Drawing.Point(12, 217);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(124, 68);
            this.button3.TabIndex = 11;
            this.button3.Text = "Sharpen";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click_1);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1196, 596);
            this.Controls.Add(this.button3);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.pictureBox2);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.manualbtn);
            this.Controls.Add(this.badbtn);
            this.Controls.Add(this.goodbtn);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.uploadbtn);
            this.Name = "Form1";
            this.Text = "Image classification tool";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.Button uploadbtn;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Button goodbtn;
        private System.Windows.Forms.Button badbtn;
        private System.Windows.Forms.Button manualbtn;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.PictureBox pictureBox2;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Button button3;
    }
}

