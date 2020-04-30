using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            //GenerateIdList();

        }
        public static ImageList LoadImages()
        {
            using (StreamReader r = new StreamReader("file.json"))
            {
                string json = r.ReadToEnd();
                Console.Write(json);
                ImageList items = JsonConvert.DeserializeObject<ImageList>(json);
                return items;
            }
        }

        public static void WriteImageId(List<Pair> pairs)
        {
            //open file stream
            using (StreamWriter file = File.CreateText(@"D:\pairs.json"))
            {
                JsonSerializer serializer = new JsonSerializer();
                //serialize object directly into file stream
                serializer.Serialize(file, pairs);
            }
        }
        public static void GenerateIdList()
        {
            ImageList items = LoadImages();
            List<Pair> pairs = new List<Pair>();
            foreach (Image img in items.images)
            {
                Console.WriteLine("{0}, {1}", img.file_name, img.id);
                pairs.Add(new Pair(img.id, img.file_name));
            }
            WriteImageId(pairs);
        }
    }
    public class ImageList
    {
        public List<Image> images;
    }
    public class Image
    {
        public int licence;
        public String file_name;
        public String coco_url;
        public int height;
        public int width;
        public DateTime date_captured;
        public int id;
    }

    public class Pair
    {
        public String file_name;
        public int id;

        public Pair(int id, String file_name)
        {
            this.id = id;
            this.file_name = file_name;
        }
    }
}


