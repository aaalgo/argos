#ifndef ARGOS_NODE_IMAGE
#define ARGOS_NODE_IMAGE

#include <boost/lexical_cast.hpp>
#include <jpeglib.h>
#include <opencv/cv.h>


namespace argos {
    namespace image {

        using namespace boost;
        typedef cv::Mat Image;

        static void decompress_error_exit (j_common_ptr ptr) {
            j_decompress_ptr cinfo = reinterpret_cast<j_decompress_ptr>(ptr);
            char msg[JMSG_LENGTH_MAX];
            (*cinfo->err->format_message)(ptr, msg);
            jpeg_destroy_decompress(cinfo);
            LOG(error) << msg;
            throw runtime_error(msg);
        }

        Image imread_jpeg (const std::string &path) {
            struct jpeg_decompress_struct cinfo;
            struct jpeg_error_mgr jerr;
            cinfo.err = jpeg_std_error(&jerr);
            jerr.error_exit = decompress_error_exit;
            jpeg_create_decompress(&cinfo);

            FILE *infile = fopen(path.c_str(), "rb");
            if (infile == NULL) throw runtime_error("cannot open file: " + path);

            jpeg_stdio_src(&cinfo, infile);

            jpeg_read_header(&cinfo, TRUE);
            // if only meta data is needed, goto  shortcut

            //cinfo.out_color_space
            // set out_color_space

            jpeg_start_decompress(&cinfo);

            if ((cinfo.output_components != 3) && (cinfo.output_components != 1)) throw runtime_error("must be color or gray image");

            LOG(trace) << "READ " << path;
            Image image(cinfo.output_height, cinfo.output_width, CV_8UC3);

            JSAMPROW row_pointer[1] = {reinterpret_cast<unsigned char *>(image.data)};
            while (cinfo.output_scanline < cinfo.output_height) {
                jpeg_read_scanlines(&cinfo,row_pointer,1);
                if (cinfo.output_components == 1) {
                    JSAMPROW p = row_pointer[0];
                    unsigned i = image.cols - 1;
                    for (;;) {
                        unsigned i3 = i * 3;
                        p[i3] = p[i3+1] = p[i3+2] = p[i];
                        if (i == 0) break;
                        --i;
                    }
                }
                row_pointer[0] += image.step[0];
            }
            jpeg_finish_decompress(&cinfo);
            jpeg_destroy_decompress(&cinfo);
            fclose(infile);
            return image;
        }

        static void compress_error_exit (j_common_ptr ptr) {
            j_compress_ptr cinfo = reinterpret_cast<j_compress_ptr>(ptr);
            char msg[JMSG_LENGTH_MAX];
            (*cinfo->err->format_message) (ptr, msg);
            cinfo->dest->term_destination(cinfo);
            void **buf = reinterpret_cast<void **>(cinfo->client_data);
            if (*buf) free(*buf);
            jpeg_destroy_compress(cinfo);
            LOG(error) << msg;
            throw runtime_error(msg);
        }

        void EncodeJPEG (double const *data,
                         unsigned width, unsigned height, unsigned channel,
                         std::string *buffer,
                         int quality = 80) {
            BOOST_VERIFY(sizeof(JSAMPLE) == sizeof(unsigned char));
            J_COLOR_SPACE colortype=JCS_RGB;
            std::vector<unsigned char> buf;
            BOOST_VERIFY(channel == 3);
            unsigned row_stride = 0;
            {
                BOOST_VERIFY(channel == 3);
                row_stride = width * channel;
                buf.resize(width * height * channel);
                colortype = JCS_RGB;
                for (unsigned i = 0; i < buf.size(); ++i) {
                    int v = data[i] * 255;
                    if (v < 0) v = 0;
                    if (v > 255) v = 255;
                    buf[i] = v;
                }
            }
            /*
            switch (img.spectrum()) {
                
                case 1: {
                            buf.resize(img.width() * img.height());
                            colortype = JCS_GRAYSCALE;
                            row_stride = img.width();
                            unsigned char *buf2 = &buf[0];
                            const unsigned char *ptr_g = img.data();
                            cimg_foroff(img,off) *(buf2++) = (JOCTET)*(ptr_g++);
                        }
                        break;
                case 3: {
                            buf.resize(img.width() * img.height() * 3);
                            colortype = JCS_RGB;
                            row_stride = img.width() * 3;
                            unsigned char *buf2 = &buf[0];
                            const unsigned char 
                                *ptr_r = img.data(0,0,0,0),
                                *ptr_g = img.data(0,0,0,1),
                                *ptr_b = img.data(0,0,0,2);
                            cimg_forXY(img,x,y) {
                              *(buf2++) = *(ptr_r++);
                              *(buf2++) = *(ptr_g++);
                              *(buf2++) = *(ptr_b++);
                            }
                        }
                        break;
                default:
                        throw ImageDecodingException("colorspace not supported");
            }
            */
            struct jpeg_compress_struct cinfo;
            struct jpeg_error_mgr jerr;
            cinfo.err = jpeg_std_error(&jerr);
            jerr.error_exit = compress_error_exit;
            jpeg_create_compress(&cinfo);

            unsigned char *out_buf = 0;
            long unsigned out_size = 0;

            jpeg_mem_dest(&cinfo, &out_buf, &out_size);
            cinfo.client_data = &out_buf;
            cinfo.image_width = width;
            cinfo.image_height = height;
            cinfo.input_components = channel;
            cinfo.in_color_space = colortype;
            jpeg_set_defaults(&cinfo);
            jpeg_set_quality(&cinfo, quality, TRUE);

            jpeg_start_compress(&cinfo,TRUE);

            //const unsigned int row_stride = width()*dimbuf;
            unsigned char* row_pointer[1];

            while (cinfo.next_scanline < cinfo.image_height) {
                row_pointer[0] = &buf[cinfo.next_scanline*row_stride];
                jpeg_write_scanlines(&cinfo,row_pointer,1);
            }
            jpeg_finish_compress(&cinfo);
            jpeg_destroy_compress(&cinfo);

            buffer->assign(reinterpret_cast<char *>(out_buf), out_size);
            free(out_buf);
        }

        class ImageTap: public Node {
            core::ArrayNode const *m_input;
            string m_dir;
        public:
            ImageTap (Model *model, Config const &config): Node(model, config),
                m_input(findInputAndAdd<core::ArrayNode>("input", "input")),
                m_dir(config.get<string>("dir"))
            {
                BOOST_VERIFY(m_input);
            }

            void predict () {
                //if (mode() == MODE_PREDICT) {
                    Array<double> const &array = m_input->data();
                    vector<size_t> sz;
                    array.size(&sz);
                    double const *data = array.addr();
                    for (unsigned i = 0; i < sz[0]; ++i) {
                        string jpeg;
                        EncodeJPEG(data, sz[2], sz[1], sz[3], &jpeg);
                        string path = m_dir + "/" + lexical_cast<string>(i) + ".jpg";
                        LOG(trace) << "SAVING " << path;
                        ofstream os(path.c_str(), ios::binary);
                        os.write(&jpeg[0], jpeg.size());
                        os.close();
                        data = array.walk<0>(data);
                    }
                }
            //}
        };

        class ImageNode: public Node {
        protected:
            vector<Image> m_images;
        public:
            ImageNode (Model *model, Config const &config) : Node(model, config) {
            }
            vector<Image> const &images () const {
                return m_images;
            }
        };

        class ImageInputNode: public ImageNode, public role::LabelInput<int>, public role::BatchInput {
            vector<pair<int,string>> m_paths;
            // labels of the batch
            vector<int> m_labels;
        public:
            ImageInputNode (Model *model, Config const &config) 
                : ImageNode(model, config)
            {
                string path;
                if (mode() == MODE_PREDICT) {
                    path = config.get<string>("test");
                }
                else {
                    path = config.get<string>("train");
                }
                {
                    LOG(info) << "loading image paths from " << path;
                    ifstream is(path.c_str());
                    int l;
                    string p;
                    while (is >> l >> p) {
                        m_paths.push_back(make_pair(l, std::move(p)));
                    }
                }
                role::BatchInput::init(getConfig<unsigned>("batch", "argos.global.batch"), m_paths.size(), mode());
            }

            void predict () {
                m_labels.clear();
                m_images.clear();
                LOG(trace) << "LOADING IMAGES...";
                role::BatchInput::next([this](unsigned i) {
                        m_labels.push_back(m_paths[i].first);
                        m_images.push_back(imread_jpeg(m_paths[i].second));
                });
            }

            virtual vector<int> const &labels () const {
                return m_labels;
            }
        };

        class ImageSampleNode: public core::ArrayNode {
            ImageInputNode *m_input;
            unsigned m_height;
            unsigned m_width;
            unsigned m_mirror;
            unsigned m_fix;
        public:
            ImageSampleNode (Model *model, Config const &config) 
                : core::ArrayNode(model, config),
                m_input(findInputAndAdd<ImageInputNode>("input", "input")),
                m_height(config.get<unsigned>("height", 227)),
                m_width(config.get<unsigned>("width", 227)),
                m_mirror(config.get<unsigned>("mirror", 1)),
                m_fix(config.get<unsigned>("fix", 0))
            {
                BOOST_VERIFY(m_input);
                vector<size_t> size{m_input->batch(), m_width, m_height, 3};
                resize(size);
                setType(IMAGE);
            }

            void predict () {
                vector<Image> const &images = m_input->images();
                double *out = data().addr();
                for (unsigned i = 0; i < images.size(); ++i) {
                    Image const &image = images[i];
                    unsigned yoff = rand() % (image.rows - m_height + 1);
                    unsigned xoff = rand() % (image.cols - m_width + 1);
                    if (m_fix) {
                        yoff = 0;
                        xoff = 0;
                    }
                    unsigned char const *p = image.data;
                    p += yoff * image.step[0];
                    p += xoff * 3;
                    for (unsigned y = 0; y < m_height; ++y) {
                        unsigned char const *r = p;
                        for (unsigned x = 0; x < m_width; ++x) {
                            out[0] = r[0] / 255.0;
                            out[1] = r[1] / 255.0;
                            out[2] = r[2] / 255.0;
                            out += 3;
                            r += 3;
                        }
                        p += image.step[0];
                    }
                }
                BOOST_VERIFY(out == data().addr() + data().size());
            }
        };

    }
}

#endif

